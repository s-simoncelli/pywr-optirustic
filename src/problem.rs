use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Formatter};
use std::path::Path;

use crate::error::WrapperError;
use crate::scenario::{ConstraintConfig, OptimisationScenario};
use log::{debug, info, warn};
use optirustic::core::{
    BoundedNumber, Constraint, EvaluationResult, Evaluator, Individual, Objective, VariableType,
};
use pywr_core::models::Model;
use pywr_core::parameters::{
    ActivationFunction, ParameterName, RbfProfileVariableConfig, VariableConfig,
};
use pywr_core::solvers::{ClpSolver, ClpSolverSettings};
use pywr_schema::outputs::Output;
use pywr_schema::parameters::{
    Parameter as SchemaParameter, Parameter, ParameterType, VariableSettings,
};
use pywr_schema::PywrModel;

/// The type of variable parameter.
#[derive(Debug, PartialEq)]
enum VariableParameterType {
    RbfProfile,
    Constant,
    Offset,
}

/// Map the parameter from the schema
impl TryFrom<&SchemaParameter> for VariableParameterType {
    type Error = WrapperError;

    fn try_from(value: &SchemaParameter) -> Result<Self, Self::Error> {
        match value {
            Parameter::Constant(_) => Ok(VariableParameterType::Constant),
            Parameter::Offset(_) => Ok(VariableParameterType::Offset),
            Parameter::RbfProfile(_) => Ok(VariableParameterType::RbfProfile),
            _ => Err(WrapperError::NotSupportedParameterType(
                value.parameter_type().to_string(),
            )),
        }
    }
}

/// Struct containing the configuration for a variable parameter. This is used
/// to set parameter values for a new generation.
struct VariableParameterConfig {
    /// The parameter name
    name: String,
    /// The struct implementing the pywr `VariableConfig` trait to handle the optimisation
    /// variable.
    variable_config: Box<dyn VariableConfig>,
    /// The optirustic optimisation variables used by the parameter.
    variables: Vec<VariableType>,
    /// The type of variable parameter being optimised.
    r#type: VariableParameterType,
}

impl Debug for VariableParameterConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VariableParameterConfig")
            .field("name", &self.name)
            .field("variables", &self.variables)
            .field("type", &self.r#type)
            .finish()
    }
}

impl VariableParameterConfig {
    /// Add one variable from a parameter supporting `f64` (for example Constant or Offset).
    ///
    /// # Arguments
    ///
    /// * `name`: The parameter name.
    /// * `parameter_type`: The parameter type.
    /// * `variable_config`: The configuration of the variable parameter.
    ///
    /// returns: `F64ParameterResult`
    fn add_f64_parameter_variable(
        name: &str,
        parameter_type: &ParameterType,
        variable_config: &Option<VariableSettings>,
    ) -> F64ParameterResult {
        if let Some(variable_config) = variable_config {
            let parameter_type = format!("{parameter_type}").to_lowercase();
            if !variable_config.is_active {
                warn!("The {parameter_type} parameter '{name}' is setup to be variable, but its configuration is not active. Skipping");
                return Ok(None);
            }

            let activation_function: ActivationFunction = variable_config.activation.into();
            let lb = activation_function.lower_bound();
            let ub = activation_function.upper_bound();
            info!("Setting 1 variable in {parameter_type} parameter '{name}':");
            info!("\t- '{name}' (real) bounded to [{lb}, {ub}]");

            let var_name = Self::get_f64_value_var(name);
            Ok(Some((
                VariableType::Real(BoundedNumber::new(&var_name, lb, ub)?),
                Box::new(activation_function),
            )))
        } else {
            Ok(None)
        }
    }

    /// Get the variable name of an x-value of Radial Basic Function. The name depends on the
    /// parameter name and the point index.
    ///
    /// # Arguments
    ///
    /// * `parameter_name`: The parameter name.
    /// * `point_index`: The point index.
    ///
    /// returns: `String`
    fn get_rbf_day_var(parameter_name: &str, point_index: usize) -> String {
        format!("{parameter_name} - Day #{}", point_index + 1)
    }

    /// Get the variable name of a y-value of Radial Basic Function. The name depends on the
    /// parameter name and the point index.
    ///
    /// # Arguments
    ///
    /// * `parameter_name`: The parameter name.
    /// * `point_index`: The point index.
    ///
    /// returns: `String`
    fn get_rbf_value_var(parameter_name: &str, var_index: usize) -> String {
        format!("{parameter_name} - Value #{}", var_index + 1)
    }

    /// get the variable name for a parameter using one `f64` variable (such as the Constant or
    /// Offset parameter).
    ///
    /// # Arguments
    ///
    /// * `parameter_name`:  The parameter name.
    ///
    /// returns: `String`
    fn get_f64_value_var(parameter_name: &str) -> String {
        parameter_name.to_string()
    }

    /// Create a `VariableParameterConfig` instance from a parameter from a pywr schema. This
    /// returns `None`
    ///
    /// # Arguments
    ///
    /// * `schema_parameter`: The pywr parameter schema.
    ///
    /// returns: `Result<Option<VariableParameterConfig>, OError>`
    fn from_schema(schema_parameter: &SchemaParameter) -> Result<Option<Self>, WrapperError> {
        let parameter_name = schema_parameter.name();
        let parameter_type = schema_parameter.parameter_type();
        let p_type = schema_parameter.try_into()?;

        match schema_parameter {
            SchemaParameter::Constant(constant) => {
                let opt_variable = Self::add_f64_parameter_variable(
                    parameter_name,
                    &parameter_type,
                    &constant.variable,
                )?;
                if let Some((variable, variable_config)) = opt_variable {
                    Ok(Some(VariableParameterConfig {
                        name: parameter_name.to_string(),
                        variable_config,
                        variables: vec![variable],
                        r#type: p_type,
                    }))
                } else {
                    Ok(None)
                }
            }
            SchemaParameter::Offset(offset) => {
                let opt_variable = Self::add_f64_parameter_variable(
                    parameter_name,
                    &parameter_type,
                    &offset.variable,
                )?;
                if let Some((variable, variable_config)) = opt_variable {
                    Ok(Some(VariableParameterConfig {
                        name: parameter_name.to_string(),
                        variable_config,
                        variables: vec![variable],
                        r#type: p_type,
                    }))
                } else {
                    Ok(None)
                }
            }
            SchemaParameter::RbfProfile(rbf) => {
                if let Some(variable_config) = &rbf.variable {
                    if !variable_config.is_active {
                        warn!("The RBF parameter '{parameter_name}' is setup to be variable, but its configuration is not active. Skipping");
                        Ok(None)
                    } else {
                        let total_points = rbf.points.len();
                        if total_points == 0 {
                            return Err(WrapperError::EmptyRBFProfile(parameter_name.to_string()));
                        }

                        // check that the first value is 1
                        if rbf.points[0].0 != 1 {
                            return Err(WrapperError::RbfNoDay1(parameter_name.to_string()));
                        }

                        info!("Setting variables in RBF profile parameter '{parameter_name}':");
                        let var_config: RbfProfileVariableConfig = (*variable_config).into();
                        let day_range = var_config.days_of_year_range();

                        // check if the day range is valid
                        if let Some(day_range) = day_range {
                            if day_range >= 364 {
                                return Err(WrapperError::InvalidParameterValueBounds(
                                    parameter_name.to_string(),
                                    "the day range must be a number smaller than 364".to_string(),
                                ));
                            }

                            let days: Vec<u32> = rbf.points.iter().map(|p| p.0).collect();
                            let day_distances: Vec<u32> = days
                                .windows(2)
                                .map(|window| window[1] - window[0])
                                .collect();

                            if day_distances.iter().any(|d| d <= &(2 * day_range)) {
                                return Err(WrapperError::InvalidParameterValueBounds(
                                    parameter_name.to_string(),
                                    format!("The days of the year are too close together for the given \
                                     `days_of_year_range`. This could cause the optimised days \
                                     of the year to overlap and become out of order. Either increase the \
                                     spacing of the days of the year or reduce `days_of_year_range` to \
                                     less than half the closest distance between the days of the year. \
                                     Current distances are {:?}", day_distances),
                                ));
                            }
                        }
                        let value_lb = var_config.value_lower_bounds();
                        let value_ub = var_config.value_upper_bounds();

                        let mut variables = vec![];
                        for (var_index, point) in rbf.points.iter().enumerate() {
                            // do not optimise day if the range is None
                            if let Some(day_range) = day_range {
                                // day variables - skip day 1 as this is not optimised
                                if var_index > 0 {
                                    let var_name = Self::get_rbf_day_var(parameter_name, var_index);
                                    let lb = (point.0 - day_range) as i64;
                                    let ub = (point.0 + day_range) as i64;
                                    info!(
                                        "\t- '{var_name}' (integer): {} bounded to [{lb}, {ub}]",
                                        point.0
                                    );
                                    variables.push(VariableType::Integer(BoundedNumber::new(
                                        &var_name, lb, ub,
                                    )?));
                                }
                            }

                            // add values
                            let var_name = Self::get_rbf_value_var(parameter_name, var_index);
                            info!("\t- '{var_name}' (real) bounded to [{value_lb}, {value_ub}]");

                            variables.push(VariableType::Real(BoundedNumber::new(
                                &var_name, value_lb, value_ub,
                            )?));
                        }

                        Ok(Some(VariableParameterConfig {
                            name: parameter_name.to_string(),
                            variable_config: Box::new(*variable_config),
                            variables,
                            r#type: p_type,
                        }))
                    }
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}

/// Struct containing the optimisation problem and its data.
pub(crate) struct PywrProblem {
    /// The pywr model.
    model: Model,
    /// The model schema used in the optimisation problem.
    pub(crate) schema: PywrModel,
    /// Map containing the parameter name and its variable configuration.
    variable_configs: HashMap<String, VariableParameterConfig>,
    /// The list of objectives.
    pub(crate) objectives: Vec<Objective>,
    /// The list of constraints.
    pub(crate) constraints: Vec<Constraint>,
}

impl Debug for PywrProblem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PywrProblem")
            .field("variable_configs", &self.variable_configs)
            .field("objectives", &self.objectives)
            .finish()
    }
}

/// Result type for add_f64_parameter_variable()
type F64ParameterResult = Result<Option<(VariableType, Box<dyn VariableConfig>)>, WrapperError>;

impl PywrProblem {
    /// Define and set up a new pywr optimisation problem.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to the JSON file.
    /// * `data_path`: The path where the model data is stored.
    /// * `scenario`: The optimisation scenario used to define the objectives and constraints.
    ///
    /// returns: `Result<PywrProblem, WrapperError>`
    pub(crate) fn new(
        path: &Path,
        data_path: Option<&Path>,
        scenario: OptimisationScenario,
    ) -> Result<Self, WrapperError> {
        // load the model file
        let data =
            std::fs::read_to_string(path).map_err(|e| WrapperError::Generic(e.to_string()))?;
        let data_path = data_path.or_else(|| path.parent());

        Self::new_from_str(data.as_str(), data_path, scenario)
    }

    /// Define and set up a new pywr optimisation problem from a model provided as string.
    ///
    /// # Arguments
    ///
    /// * `data`: The model as string.
    /// * `data_path`: The path where the model data is stored. Optional.
    /// * `scenario`: The optimisation scenario used to define the objectives and constraints.
    ///
    /// returns: `Result<PywrProblem, WrapperError>`
    pub(crate) fn new_from_str(
        data: &str,
        data_path: Option<&Path>,
        scenario: OptimisationScenario,
    ) -> Result<Self, WrapperError> {
        // parse the schema
        let mut schema: PywrModel =
            serde_json::from_str(data).map_err(|e| WrapperError::Pywr(e.to_string()))?;

        // add variables
        let mut variable_configs: HashMap<String, VariableParameterConfig> = HashMap::new();
        if let Some(parameters) = &mut schema.network.parameters {
            // add parameters from scenario to the schema
            if let Some(sc_parameters) = &scenario.parameters {
                for parameter in sc_parameters.iter() {
                    // check if parameter already exists
                    if let Some(item) = parameters
                        .iter_mut()
                        .find(|item| item.name() == parameter.name())
                    {
                        info!(
                            "Overwriting parameter '{}' from scenario into schema",
                            parameter.name()
                        );
                        let _ = std::mem::replace(item, parameter.clone());
                    } else {
                        info!(
                            "Adding new parameter '{}' from scenario into schema",
                            parameter.name()
                        );
                        parameters.push(parameter.clone());
                    }
                }
            }

            // scan parameters in schema to check for settings for variables
            for schema_parameter in parameters.iter() {
                debug!(
                    "Checking if '{}' parameter has variables",
                    schema_parameter.name()
                );
                let parameter_name = schema_parameter.name();

                if let Some(var_parameter_config) =
                    VariableParameterConfig::from_schema(schema_parameter)?
                {
                    variable_configs.insert(parameter_name.to_string(), var_parameter_config);
                } else {
                    debug!("Skipping; parameter is not set as variable");
                    continue;
                }
            }
            if variable_configs.is_empty() {
                return Err(WrapperError::NoVariableParameters);
            }
        } else {
            return Err(WrapperError::NoModelParameters);
        }

        // add scenario metrics before objectives and constraints
        if schema.network.metric_sets.is_none() {
            schema.network.metric_sets = Some(Vec::new());
        }
        let metric_sets = schema.network.metric_sets.as_mut().unwrap();
        // let metric_sets = &mut schema.network.metric_sets.unwrap_or_default();
        if let Some(sc_metric_sets) = &scenario.metric_sets {
            for sc_metric_set in sc_metric_sets.iter() {
                if let Some(item) = metric_sets
                    .iter_mut()
                    .find(|item| item.name == sc_metric_set.name)
                {
                    info!(
                        "Overwriting metric set '{}' from scenario into schema",
                        sc_metric_set.name
                    );
                    let _ = std::mem::replace(item, sc_metric_set.clone());
                } else {
                    info!(
                        "Adding metric set '{}' from scenario into schema",
                        sc_metric_set.name
                    );
                    metric_sets.push(sc_metric_set.clone());
                }
            }
        }

        // add scenario recorders before objectives and constraints
        if schema.network.outputs.is_none() {
            schema.network.outputs = Some(Vec::new());
        }
        let outputs = &mut schema.network.outputs.as_mut().unwrap();
        if let Some(sc_recorders) = &scenario.memory_recorders {
            for sc_recorder in sc_recorders.iter() {
                match sc_recorder {
                    Output::Memory(sc_recorder) => {
                        println!("{:?}", sc_recorder);
                        if let Some(item) = outputs.iter_mut().find(|item| {
                            // match only recorders of same type. If recorder has same name
                            // but different type, always replace it
                            match item {
                                Output::Memory(item) => item.name == sc_recorder.name,
                                _ => true,
                            }
                        }) {
                            info!(
                                "Overwriting output '{}' from scenario into schema",
                                sc_recorder.name
                            );
                            let _ = std::mem::replace(item, Output::Memory(sc_recorder.clone()));
                        } else {
                            info!(
                                "Adding output '{}' from scenario into schema",
                                sc_recorder.name
                            );
                            outputs.push(Output::Memory(sc_recorder.clone()));
                        }
                    }
                    _ => {
                        return Err(WrapperError::InvalidScenarioRecorder(
                            sc_recorder.to_string(),
                        ))
                    }
                }
            }
        }

        // load the model and its data
        let model = schema
            .build_model(data_path, None)
            .map_err(|e| WrapperError::Pywr(e.to_string()))?;

        // collect objectives
        let mut objectives: Vec<Objective> = vec![];
        for sc_objective in scenario.objectives.iter() {
            // check that the recorder exists
            debug!(
                "Checking if recorder '{:?}' exists for objective",
                sc_objective.recorder_name()
            );
            if !Self::does_recorder_exist(&model, sc_objective.recorder_name()) {
                return Err(WrapperError::MissingRecorder(
                    sc_objective.recorder_name().to_string(),
                ));
            }
            let objective = sc_objective.to_opt_constraint();
            info!(
                "Setting objective '{}' on recorder '{}' with direction {}",
                objective.name(),
                sc_objective.recorder_name(),
                sc_objective.direction()
            );
            objectives.push(objective);
        }

        // collect constraints
        let mut constraints: Vec<Constraint> = vec![];
        if let Some(sc_constraints) = &scenario.constraints {
            for sc_constraint in sc_constraints.iter() {
                debug!(
                    "Checking if recorder '{}' exists for constraint",
                    sc_constraint.recorder_name()
                );
                // check that the recorder exists
                if !Self::does_recorder_exist(&model, sc_constraint.recorder_name()) {
                    return Err(WrapperError::MissingRecorder(
                        sc_constraint.recorder_name().to_string(),
                    ));
                }
                constraints.extend(sc_constraint.to_opt_constraint());
                info!(
                    "Setting constraints on recorder '{}' to {}",
                    sc_constraint.recorder_name(),
                    sc_constraint.get_bound_string()
                );
            }
        } else {
            warn!("No constraints have been defined");
        }

        Ok(PywrProblem {
            model,
            schema,
            variable_configs,
            objectives,
            constraints,
        })
    }

    /// Check that a recorder exists.
    ///
    /// # Arguments
    ///
    /// * `model`: The pywr model.
    /// * `name`: The name of the recorder.
    ///
    /// returns: `bool`
    fn does_recorder_exist(model: &Model, name: &str) -> bool {
        model.network().recorders().iter().any(|r| r.name() == name)
    }

    /// Get the optirustic variables.
    ///
    /// returns: `Vec<VariableType>`
    pub(crate) fn variables(&self) -> Vec<VariableType> {
        let mut variables: Vec<VariableType> = vec![];
        for data in self.variable_configs.values() {
            for var in data.variables.iter() {
                variables.push(var.clone());
            }
        }
        variables
    }

    // /// Get the optirustic objectives.
    // ///
    // /// returns: `Vec<Objective>`
    // pub fn objectives(&self) -> &Vec<Objective> {
    //     &self.objectives
    // }
    //
    // /// Get the optirustic constraints.
    // ///
    // /// returns: `Vec<Constraint>`
    // pub fn constraints(&self) -> &Vec<Constraint> {
    //     &self.constraints
    // }

    /// Get a vector containing the variable values for a parameter from an individual. This is used
    /// to get the variable values chosen by the genetic algorithm and stored in an `Individual`.
    /// The values can then be set on the `parameter_name` to calculate the new problem objectives
    /// and constraints.
    ///
    /// This returns an error if a variable cannot be found for the parameter or its values is not
    /// a `f64`.
    /// # Arguments
    ///
    /// * `parameter_name`: The parameter
    /// * `individual`: The individual with the variables set by a genetic algorithm.
    ///
    /// returns: `Result<Vec<f64>, WrapperError>`
    pub(crate) fn get_f64_variable_vector(
        &self,
        parameter_name: &ParameterName,
        individual: &Individual,
    ) -> Result<Vec<f64>, WrapperError> {
        let parameter_string: String = format!("{parameter_name}");
        if let Some(data) = self.variable_configs.get(&parameter_string) {
            let mut values: Vec<f64> = vec![];
            match data.r#type {
                VariableParameterType::RbfProfile => {
                    // the first value is always 1 and not optimised
                    values.push(1.0);

                    // append day values from point #2
                    let rbf_size = (data.variables.len() + 1) / 2;
                    for var_index in 1..rbf_size {
                        let var_name =
                            VariableParameterConfig::get_rbf_day_var(&parameter_string, var_index);
                        let value = individual.get_variable_value(&var_name)?.as_integer()?;
                        values.push(value as f64);
                    }
                    // append y-values
                    for var_index in 0..rbf_size {
                        let var_name = VariableParameterConfig::get_rbf_value_var(
                            &parameter_string,
                            var_index,
                        );
                        let value = individual.get_variable_value(&var_name)?.as_real()?;
                        values.push(value);
                    }
                }
                VariableParameterType::Constant | VariableParameterType::Offset => {
                    let var_name = VariableParameterConfig::get_f64_value_var(&parameter_string);
                    let value = individual.get_variable_value(&var_name)?.as_real()?;
                    values.push(value);
                }
            }
            Ok(values)
        } else {
            Err(WrapperError::NotSupportedParameterType(
                parameter_name.to_string(),
            ))
        }
    }
}

impl Evaluator for PywrProblem {
    fn evaluate(&self, individual: &Individual) -> Result<EvaluationResult, Box<dyn Error>> {
        // load the model state
        let mut state = self
            .model
            .setup::<ClpSolver>(&ClpSolverSettings::default())?;
        let network = self.model.network();

        // configure the parameter using the variables from the individual
        for (param_name, variable_config) in self.variable_configs.iter() {
            // NOTE: model parameters are built w/o parent as name
            let name: ParameterName = param_name.as_str().into();
            let parameter_index = network.get_parameter_index_by_name(&name)?;

            // fetch the variables in the individuals linked to this parameter
            let vars = self.get_f64_variable_vector(&name, individual)?;
            network.set_f64_parameter_variable_values(
                parameter_index,
                &vars,
                variable_config,
                state.network_state_mut(),
            )?;
        }

        // run model
        self.model
            .run_with_state(&mut state, &ClpSolverSettings::default())?;

        // collect the objectives
        let mut objectives: HashMap<String, f64> = HashMap::new();
        for objective in self.objectives.iter() {
            // the recorder name is the objective name
            let rec_name = objective.name();
            let rec_value = self
                .model
                .network()
                .get_aggregated_value(&rec_name, state.recorder_state())?;
            objectives.insert(objective.name(), rec_value);
        }

        // collect the constraints
        let mut constraints: HashMap<String, f64> = HashMap::new();
        for constraint in self.constraints.iter() {
            let rec_name = ConstraintConfig::get_recorder_name(constraint);
            // check if recorder's value was already calculated as objective
            let rec_value = if objectives.contains_key(&rec_name) {
                objectives[&rec_name]
            } else {
                self.model
                    .network()
                    .get_aggregated_value(&rec_name, state.recorder_state())?
            };
            constraints.insert(constraint.name(), rec_value);
        }

        Ok(EvaluationResult {
            constraints: Some(constraints),
            objectives,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::{PywrProblem, VariableParameterConfig, VariableParameterType};
    use crate::scenario::{Bound, ConstraintConfig, ObjectiveConfig, OptimisationScenario};
    use optirustic::core::utils::dummy_evaluator;
    use optirustic::core::{Individual, Objective, ObjectiveDirection, Problem, VariableValue};
    use pywr_core::parameters::ParameterName;
    use std::env;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    fn test_path() -> PathBuf {
        Path::new(&env::current_dir().unwrap())
            .join("src")
            .join("test_data")
    }

    fn dummy_scenario() -> OptimisationScenario {
        OptimisationScenario {
            name: "Test scenario".to_string(),
            description: None,
            objectives: vec![ObjectiveConfig::new(
                "outputs".to_string(),
                ObjectiveDirection::Minimise,
                None,
            )],
            constraints: Some(vec![ConstraintConfig::new(
                "outputs".to_string(),
                Some(Bound::new(10.0, true)),
                None,
                None,
            )
            .unwrap()]),
            parameters: None,
            metric_sets: None,
            memory_recorders: None,
        }
    }

    /// Test the variable initialisation for a constant parameter.
    #[test]
    fn test_constant_parameter() {
        let file = test_path().join("constant_var_parameter.json");
        let pywr_problem = PywrProblem::new(&file, None, dummy_scenario()).unwrap();
        let data = pywr_problem.variable_configs.get("demand").unwrap();

        assert_eq!(data.name, "demand");
        assert_eq!(data.r#type, VariableParameterType::Constant);
        assert_eq!(data.variables.len(), 1);
        assert_eq!(data.variables[0].name(), "demand");
        assert_eq!(data.variables[0].label(), "real");

        // check objectives
        assert_eq!(pywr_problem.objectives.len(), 1);
        assert_eq!(
            pywr_problem.objectives[0].name(),
            "'outputs' objective".to_string()
        );
        assert_eq!(
            pywr_problem.objectives[0].direction(),
            ObjectiveDirection::Minimise
        );

        // check constraints
        assert_eq!(pywr_problem.constraints.len(), 1);
        assert_eq!(
            format!("{}", pywr_problem.constraints[0]),
            "'outputs' lower bound > 10".to_string()
        );

        let objectives = &pywr_problem.objectives;
        let variables = pywr_problem.variables();
        // use dummy evaluator not to move pywr_problem
        let problem = Problem::new(objectives.clone(), variables, None, dummy_evaluator()).unwrap();

        let mut dummy_individual = Individual::new(Arc::new(problem));
        dummy_individual
            .update_variable("demand", VariableValue::Real(19.5))
            .unwrap();

        assert_eq!(
            pywr_problem
                .get_f64_variable_vector(&ParameterName::new("demand", None), &dummy_individual)
                .unwrap(),
            vec![19.5]
        );
    }

    /// Test the variable initialisation for an RBF parameter.
    #[test]
    fn test_rbf_parameter() {
        let file = test_path().join("rbf_var_parameter.json");
        let pywr_problem = PywrProblem::new(&file, None, dummy_scenario()).unwrap();
        let data = pywr_problem.variable_configs.get("demand").unwrap();

        // use dummy evaluator not to move pywr_problem
        let objectives = vec![Objective::new(
            "dummy_objective",
            ObjectiveDirection::Minimise,
        )];
        let variables = pywr_problem.variables();
        let problem = Problem::new(objectives, variables, None, dummy_evaluator()).unwrap();
        let mut dummy_individual = Individual::new(Arc::new(problem));

        assert_eq!(data.name, "demand");
        assert_eq!(data.r#type, VariableParameterType::RbfProfile);
        assert_eq!(data.variables.len(), 5);

        // X1 does not exist
        // Y1
        let var_name = VariableParameterConfig::get_rbf_value_var("demand", 0);
        assert_eq!(data.variables[0].name(), var_name);
        assert_eq!(data.variables[0].label(), "real");
        dummy_individual
            .update_variable(&var_name, VariableValue::Real(21.0))
            .unwrap();

        // X2
        let var_name = VariableParameterConfig::get_rbf_day_var("demand", 1);
        assert_eq!(data.variables[1].name(), var_name);
        assert_eq!(data.variables[1].label(), "integer");
        assert_eq!(data.variables[1].label(), "integer");
        dummy_individual
            .update_variable(&var_name, VariableValue::Integer(45))
            .unwrap();

        // Y2
        let var_name = VariableParameterConfig::get_rbf_value_var("demand", 1);
        assert_eq!(data.variables[2].name(), var_name);
        assert_eq!(data.variables[2].label(), "real");
        dummy_individual
            .update_variable(&var_name, VariableValue::Real(67.1))
            .unwrap();

        // X3
        let var_name = VariableParameterConfig::get_rbf_day_var("demand", 2);
        assert_eq!(data.variables[3].name(), var_name);
        assert_eq!(data.variables[3].label(), "integer");
        dummy_individual
            .update_variable(&var_name, VariableValue::Integer(210))
            .unwrap();

        // Y3
        let var_name = VariableParameterConfig::get_rbf_value_var("demand", 2);
        assert_eq!(data.variables[4].name(), var_name);
        assert_eq!(data.variables[4].label(), "real");
        dummy_individual
            .update_variable(&var_name, VariableValue::Real(13.34))
            .unwrap();

        assert_eq!(
            pywr_problem
                .get_f64_variable_vector(&ParameterName::new("demand", None), &dummy_individual)
                .unwrap(),
            vec![1.0, 45.0, 210.0, 21.0, 67.1, 13.34]
        );
    }

    /// The RBF profile has no points and its size cannot be determined
    #[test]
    fn test_empty_rbf_profile() {
        let file = test_path().join("empty_rbf_var_parameter.json");
        let pywr_problem = PywrProblem::new(&file, None, dummy_scenario());
        assert!(pywr_problem
            .err()
            .unwrap()
            .to_string()
            .contains("because the property 'points' is empty"));
    }

    /// The first x-value in the RBF profile must be 1.
    #[test]
    fn test_rbf_profile_day_1() {
        let file = test_path().join("rbf_var_parameter_no_day_1.json");
        let pywr_problem = PywrProblem::new(&file, None, dummy_scenario());
        assert!(pywr_problem
            .err()
            .unwrap()
            .to_string()
            .contains("The first x-value in the RBF parameter"));
    }

    /// The RBF profile has a day range larger than 364.
    #[test]
    fn test_invalid_rbf_profile_day_bound() {
        let file = test_path().join("rbf_var_parameter_invalid_day_bound.json");
        let pywr_problem = PywrProblem::new(&file, None, dummy_scenario());
        assert!(pywr_problem
            .err()
            .unwrap()
            .to_string()
            .contains("the day range must be a number smaller than 364"));
    }

    /// The RBF profile has variable days that are too close.
    #[test]
    fn test_invalid_rbf_profile_days_too_close() {
        let file = test_path().join("rbf_var_parameter_points_too_close.json");
        let pywr_problem = PywrProblem::new(&file, None, dummy_scenario());
        assert!(pywr_problem
            .err()
            .unwrap()
            .to_string()
            .contains("The days of the year are too close together"));
    }

    /// The model has no variable parameters.
    #[test]
    fn test_no_var_parameters() {
        let file = test_path().join("no_var_parameters.json");
        let pywr_problem = PywrProblem::new(&file, None, dummy_scenario());
        assert!(pywr_problem
            .err()
            .unwrap()
            .to_string()
            .contains("The model does not contain any variable parameters"));
    }

    /// The model has no valid recorders for the objective and constraints.
    #[test]
    fn test_no_recorder() {
        let file = test_path().join("no_recorder.json");
        let pywr_problem = PywrProblem::new(&file, None, dummy_scenario());
        assert!(pywr_problem
            .err()
            .unwrap()
            .to_string()
            .contains("The recorder named 'outputs' does not exist"));
    }

    /// The model has no parameters.
    #[test]
    fn test_no_parameters() {
        let file = test_path().join("no_parameters.json");
        let pywr_problem = PywrProblem::new(&file, None, dummy_scenario());
        assert!(pywr_problem
            .err()
            .unwrap()
            .to_string()
            .contains("does not contain any parameter"));
    }

    #[test]
    /// Check that the scenario data are added to the schema and model
    fn test_scenario() {
        let file = test_path().join("model_for_scenario.json");
        let scenario =
            OptimisationScenario::load_from_file(&test_path().join("opt_scenario.json")).unwrap();
        let pywr_problem = PywrProblem::new(&file, None, scenario).unwrap();

        assert_eq!(pywr_problem.objectives[0].name(), "outputs");
        assert_eq!(pywr_problem.constraints[0].name(), "'outputs' lower bound");
        assert_eq!(pywr_problem.constraints[0].target(), 5.0);

        // check parameter is added
        let p_name = ParameterName {
            name: "demand".to_string(),
            parent: None,
        };
        assert!(pywr_problem
            .model
            .network()
            .get_parameter_index_by_name(&p_name)
            .is_ok());

        // check metric set
        assert!(pywr_problem
            .model
            .network()
            .get_metric_set_by_name("node_metric")
            .is_ok());

        // check recorder
        assert!(pywr_problem
            .model
            .network()
            .get_recorder_by_name("outputs")
            .is_ok());
    }
}
