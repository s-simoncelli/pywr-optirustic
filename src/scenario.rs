use crate::error::WrapperError;
use optirustic::core::{Constraint, Objective, ObjectiveDirection, RelationalOperator};
use pywr_schema::metric_sets::MetricSet;
use pywr_schema::outputs::Output;
use pywr_schema::parameters::Parameter;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

/// Define an objective for the optimisation problem.
#[derive(Deserialize)]
pub struct ObjectiveConfig {
    /// The name of the memory recorder to minimise or maximise.
    recorder_name: String,
    /// The optimisation direction to define the minimisation or maximisation of the aggregated
    /// recorder value.
    direction: ObjectiveDirection,
    /// An optional comment
    comment: Option<String>,
}

impl ObjectiveConfig {
    /// Create a new scenario objective. Objectives are set on a model memory recorder which
    /// always return one number at the end of a simulation when an aggregation is specified.
    ///
    /// # Arguments
    ///
    /// * `recorder_name`: The name of the memory recorder to minimise or maximise.
    /// * `objective_direction`: The optimisation direction to define the minimisation or
    ///    maximisation of the recorder value.
    /// * `comment`: An optional comment.
    ///
    /// returns: `ObjectiveConfig`
    pub fn new(
        recorder_name: String,
        objective_direction: ObjectiveDirection,
        comment: Option<String>,
    ) -> Self {
        Self {
            recorder_name,
            direction: objective_direction,
            comment,
        }
    }

    /// Get the recorder's name.
    pub fn recorder_name(&self) -> &str {
        &self.recorder_name
    }

    /// Get the objective direction.
    pub fn direction(&self) -> &ObjectiveDirection {
        &self.direction
    }

    /// Generate am optimisation objective from the scenario objective. The optimisation
    /// objective name is the recorder's name.
    ///
    /// returns: `Objective`
    pub(crate) fn to_opt_constraint(&self) -> Objective {
        Objective::new(&self.recorder_name, self.direction)
    }
}

/// Define a constraint lower or upper bound.
#[derive(Deserialize)]
pub struct Bound {
    /// The bound value.
    value: f64,
    /// Whether to include or exclude the `value` from the relational operator.
    strict: bool,
}

impl Bound {
    /// Create a new bound.
    ///
    /// # Arguments
    ///
    /// * `value`: The bound value.
    /// * `strict`: Whether to include or exclude the `value` from the relational operator.
    ///    For example, if this is a lower bound and `strict` is `true`, the constraint on the
    ///    recorder translates to `aggregated_recorder_value > value`.
    ///
    /// returns: `Bound`
    pub fn new(value: f64, strict: bool) -> Self {
        Self { value, strict }
    }
}

/// An optimisation problem constraint.
#[derive(Deserialize)]
pub struct ConstraintConfig {
    /// The name of the aggregated memory recorder to constraint.
    recorder_name: String,
    /// The recorder value lower bound.
    lower_bound: Option<Bound>,
    /// The recorder value upper bound.
    upper_bound: Option<Bound>,
    /// An optional comment.
    comment: Option<String>,
}

impl ConstraintConfig {
    /// Create a new scenario constraint. Constraints are set on a model memory recorder which
    /// always returns one number at the end of a simulation.
    ///
    /// # Arguments
    ///
    /// * `recorder_name`: The name of the memory recorder to constraint.
    /// * `lower_bound`: The recorder value lower bound.
    /// * `upper_bound`: The recorder value upper bound.
    /// * `comment`: An optional comment.
    ///
    /// returns: `Result<ConstraintConfig, WrapperError>`
    pub fn new(
        recorder_name: String,
        lower_bound: Option<Bound>,
        upper_bound: Option<Bound>,
        comment: Option<String>,
    ) -> Result<ConstraintConfig, WrapperError> {
        if lower_bound.is_none() && upper_bound.is_none() {
            return Err(WrapperError::InvalidConstraintBounds(
                recorder_name,
                "You must provide either a lower or upper bound".to_string(),
            ));
        }
        Ok(ConstraintConfig {
            recorder_name,
            lower_bound,
            upper_bound,
            comment,
        })
    }

    /// Get the recorder's name.
    pub fn recorder_name(&self) -> &str {
        &self.recorder_name
    }

    /// Generate a vector of optimisation constraints from the scenario constraint. The optimisation
    /// constraints are named using the memory recorder value followed by the suffix 'lower bound'
    /// or 'upper bound'.
    ///
    /// returns: `Vec<Constraint>`
    pub(crate) fn to_opt_constraint(&self) -> Vec<Constraint> {
        let mut opt_constraints = vec![];
        if let Some(lower_bound) = &self.lower_bound {
            let operator = if lower_bound.strict {
                RelationalOperator::GreaterThan
            } else {
                RelationalOperator::GreaterOrEqualTo
            };
            opt_constraints.push(Constraint::new(
                &format!("'{}' lower bound", &self.recorder_name),
                operator,
                lower_bound.value,
            ));
        }

        if let Some(upper_bound) = &self.upper_bound {
            let operator = if upper_bound.strict {
                RelationalOperator::LessThan
            } else {
                RelationalOperator::LessOrEqualTo
            };
            opt_constraints.push(Constraint::new(
                &format!("'{}' upper bound", &self.recorder_name),
                operator,
                upper_bound.value,
            ));
        }

        opt_constraints
    }

    /// Get the recorder name from the constraint's name.
    ///
    /// # Arguments
    ///
    /// * `constraint`: The constraint.
    ///
    /// returns: `String`
    pub(crate) fn get_recorder_name(constraint: &Constraint) -> String {
        match constraint.operator() {
            RelationalOperator::LessOrEqualTo | RelationalOperator::LessThan => {
                let name = constraint.name().replace(" upper bound", "");
                name.replace("'", "")
            }
            RelationalOperator::GreaterOrEqualTo | RelationalOperator::GreaterThan => {
                let name = constraint.name().replace(" lower bound", "");
                name.replace("'", "")
            }
            _ => panic!("Constraint not supported"),
        }
    }

    /// Get a string showing the constraint bounds.
    ///
    /// returns: `String`
    pub(crate) fn get_bound_string(&self) -> String {
        let mut bound_string = "".to_string();

        match (&self.lower_bound, &self.upper_bound) {
            (Some(lower_bound), Some(upper_bound)) => {
                if lower_bound.strict {
                    bound_string.push(']');
                } else {
                    bound_string.push('[');
                }
                bound_string.push_str(lower_bound.value.to_string().as_str());
                bound_string.push_str("; ");

                if upper_bound.strict {
                    bound_string.push(']');
                } else {
                    bound_string.push('[');
                }
                bound_string.push_str(upper_bound.value.to_string().as_str());

                bound_string
            }
            (Some(lower_bound), None) => {
                if lower_bound.strict {
                    bound_string.push(']');
                } else {
                    bound_string.push('[');
                }
                bound_string.push_str(lower_bound.value.to_string().as_str());
                bound_string.push_str("; Inf[");
                bound_string
            }
            (None, Some(upper_bound)) => {
                bound_string.push_str("]Inf; ");

                bound_string.push_str(upper_bound.value.to_string().as_str());
                if upper_bound.strict {
                    bound_string.push(']');
                } else {
                    bound_string.push('[');
                }
                bound_string
            }
            _ => panic!("No bounds"),
        }
    }
}

/// Define an optimisation scenarios to set problem objectives and constraints
#[derive(Deserialize)]
pub struct OptimisationScenario {
    /// the scenario name.
    pub name: String,
    /// An optional description.
    pub description: Option<String>,
    /// The list of objectives.
    pub objectives: Vec<ObjectiveConfig>,
    /// The list of constraints.
    pub constraints: Option<Vec<ConstraintConfig>>,
    /// The list of variable parameters to add to the schema. Optional If the parameters are already
    /// in the model.
    pub parameters: Option<Vec<Parameter>>,
    /// The list of metric sets to use to record metrics for model components.
    pub metric_sets: Option<Vec<MetricSet>>,
    /// The list of memory recorders to use to collect metrics and set objectives and constraints.
    pub memory_recorders: Option<Vec<Output>>,
}

impl OptimisationScenario {
    /// Load the optimisation scenario from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to the JSON file.
    ///
    /// returns: `Result<OptimisationScenario, WrapperError>`
    pub fn load_from_file(path: &PathBuf) -> Result<OptimisationScenario, WrapperError> {
        let file = fs::File::open(path).map_err(|e| WrapperError::Generic(e.to_string()))?;
        let scenario: OptimisationScenario =
            serde_json::from_reader(file).map_err(|e| WrapperError::Generic(e.to_string()))?;

        Ok(scenario)
    }
}
