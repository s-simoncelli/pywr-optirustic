use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Formatter};
use std::path::Path;

use crate::error::WrapperError;
use log::{debug, info, warn};
use optirustic::core::{
    BoundedNumber, EvaluationResult, Evaluator, Individual, Objective, VariableType,
};
use pywr_core::models::Model;
use pywr_core::parameters::{
    ActivationFunction, ParameterName, RbfProfileVariableConfig, VariableConfig,
};
use pywr_core::solvers::{ClpSolver, ClpSolverSettings};
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

                        info!("Setting variables in RBF profile parameter '{parameter_name}':");
                        let var_config: RbfProfileVariableConfig = (*variable_config).into();
                        let day_range = var_config.days_of_year_range();

                        // check if the day range is valid
                        if let Some(day_range) = day_range {
                            if day_range >= 364 {
                                return Err(WrapperError::InvalidBounds(
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
                                return Err(WrapperError::InvalidBounds(
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
                                // day variables - skip day 1
                                if var_index > 0 {
                                    // TODO check distances between points
                                    // TODO check negative values
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
    /// The model schema.
    /// Map containing the parameter name and its variable configuration.
    variable_configs: HashMap<String, VariableParameterConfig>,
    /// The list of objectives.
    pub(crate) objectives: Vec<Objective>,
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
    ///
    /// returns: `Result<PywrProblem, WrapperError>`
    pub(crate) fn new(path: &Path, data_path: Option<&Path>) -> Result<Self, WrapperError> {
        // load the model file
        let data =
            std::fs::read_to_string(path).map_err(|e| WrapperError::Generic(e.to_string()))?;
        let data_path = data_path.or_else(|| path.parent());

        Self::new_from_str(data.as_str(), data_path)
    }

    /// Define and set up a new pywr optimisation problem from a model provided as string.
    ///
    /// # Arguments
    ///
    /// * `data`: The model as string.
    /// * `data_path`: The path where the model data is stored. Optional.
    ///
    /// returns: `Result<PywrProblem, WrapperError>`
    pub(crate) fn new_from_str(data: &str, data_path: Option<&Path>) -> Result<Self, WrapperError> {
        // parse the schema
        let schema: PywrModel =
            serde_json::from_str(data).map_err(|e| WrapperError::Pywr(e.to_string()))?;

        // load the model and its data
        let model = schema
            .build_model(data_path, None)
            .map_err(|e| WrapperError::Pywr(e.to_string()))?;

        // scan parameters in schema to check for settings for variables
        let mut variable_configs: HashMap<String, VariableParameterConfig> = HashMap::new();
        if let Some(parameters) = &schema.network.parameters {
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

        // collect objectives TODO
        let mut objectives: Vec<Objective> = vec![];

        Ok(PywrProblem {
            model,
            variable_configs,
            objectives,
        })
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
    pub(crate) fn get_variable_vector(
        &self,
        parameter_name: &ParameterName,
        individual: &Individual,
    ) -> Result<Vec<f64>, WrapperError> {
        let parameter_string: String = format!("{parameter_name}");
        if let Some(data) = self.variable_configs.get(&parameter_string) {
            let mut values: Vec<f64> = vec![];
            match data.r#type {
                VariableParameterType::RbfProfile => {
                    // append day values first and then values
                    let rbf_size = (data.variables.len() + 1) / 2;
                    for var_index in 1..rbf_size {
                        let var_name =
                            VariableParameterConfig::get_rbf_day_var(&parameter_string, var_index);
                        let value = individual.get_variable_value(&var_name)?.as_integer()?;
                        values.push(value as f64);
                    }
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
            let vars = self.get_variable_vector(&name, individual)?;
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
        // TODO
        let mut objectives: HashMap<String, f64> = HashMap::new();
        // objectives.insert("x^2".to_string(), SCHProblem::f1(x));

        // collect the constraints
        // TODO check if defined
        let mut constraints: HashMap<String, f64> = HashMap::new();

        Ok(EvaluationResult {
            constraints: Some(constraints),
            objectives,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::{PywrProblem, VariableParameterConfig, VariableParameterType};
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

    /// Test the variable initialisation for a constant parameter.
    #[test]
    fn test_constant_parameter() {
        let file = test_path().join("constant_var_parameter.json");
        let pywr_problem = PywrProblem::new(&file, None).unwrap();
        let data = pywr_problem.variable_configs.get("demand").unwrap();

        assert_eq!(data.name, "demand");
        assert_eq!(data.r#type, VariableParameterType::Constant);
        assert_eq!(data.variables.len(), 1);
        assert_eq!(data.variables[0].name(), "demand");
        assert_eq!(data.variables[0].label(), "real");

        let objectives = vec![Objective::new(
            "dummy_objective",
            ObjectiveDirection::Minimise,
        )];
        let variables = pywr_problem.variables();
        // use dummy evaluator not to move pywr_problem
        let problem = Problem::new(objectives, variables, None, dummy_evaluator()).unwrap();

        let mut dummy_individual = Individual::new(Arc::new(problem));
        dummy_individual
            .update_variable("demand", VariableValue::Real(19.5))
            .unwrap();

        assert_eq!(
            pywr_problem
                .get_variable_vector(&ParameterName::new("demand", None), &dummy_individual)
                .unwrap(),
            vec![19.5]
        );
    }

    /// Test the variable initialisation for an RBF parameter.
    #[test]
    fn test_rbf_parameter() {
        let file = test_path().join("rbf_var_parameter.json");
        let pywr_problem = PywrProblem::new(&file, None).unwrap();
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
                .get_variable_vector(&ParameterName::new("demand", None), &dummy_individual)
                .unwrap(),
            vec![45.0, 210.0, 21.0, 67.1, 13.34]
        );
    }

    /// The RBF profile has no points and its size cannot be determined
    #[test]
    fn test_empty_rbf_profile() {
        let file = test_path().join("empty_rbf_var_parameter.json");
        let pywr_problem = PywrProblem::new(&file, None);
        assert!(pywr_problem
            .err()
            .unwrap()
            .to_string()
            .contains("because the property 'points' is empty"));
    }

    /// The RBF profile has a day range larger than 364.
    #[test]
    fn test_invalid_rbf_profile_day_bound() {
        let file = test_path().join("rbf_var_parameter_invalid_day_bound.json");
        let pywr_problem = PywrProblem::new(&file, None);
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
        let pywr_problem = PywrProblem::new(&file, None);
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
        let pywr_problem = PywrProblem::new(&file, None);
        assert!(pywr_problem
            .err()
            .unwrap()
            .to_string()
            .contains("The model does not contain any variable parameters"));
    }

    /// The model has no parameters.
    #[test]
    fn test_no_parameters() {
        let file = test_path().join("no_parameters.json");
        let pywr_problem = PywrProblem::new(&file, None);
        assert!(pywr_problem
            .err()
            .unwrap()
            .to_string()
            .contains("does not contain any parameter"));
    }
}
