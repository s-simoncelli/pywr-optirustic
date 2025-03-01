mod error;
mod problem;
mod scenario;

use crate::error::WrapperError;
use crate::problem::PywrProblem;
use log::info;
use optirustic::algorithms::{AdaptiveNSGA3, Algorithm as OAlgorithm, ExportHistory, NSGA2, NSGA3};
use optirustic::algorithms::{NSGA2Arg, NSGA3Arg};
use optirustic::core::Problem;
use pywr_schema::PywrModel;
use std::fs;
use std::path::{Path, PathBuf};

pub use crate::scenario::{ConstraintConfig, ObjectiveConfig, OptimisationScenario};

/// The algorithm to use for the optimisation and its options.
pub enum Algorithm {
    NSGA2(NSGA2Arg),
    NSGA3(NSGA3Arg),
    AdaptiveNSGA3(NSGA3Arg),
}

/// The optirustic wrapper to solve optimisation problems using pywr
pub struct PywrOptirustic;

impl PywrOptirustic {
    /// Solve the problem with the chosen algorithm.
    ///
    /// # Arguments
    ///
    /// * `model_file`: The path to the model JSON file.
    /// * `data_path`: The optional path where the model data is stored.
    /// * `algorithm`: The algorithm and its option to use to solve the optimisation problem.
    /// * `export_history`: The data used to configure the result export for the generations. The
    ///    option in  'algorithm.export_history', if provided, is overwritten by this argument.
    /// * `scenario`: The optimisation scenario used to define the objectives and constraints.
    ///
    /// returns: `Result<(), WrapperError>`
    pub fn run(
        model_file: &Path,
        data_path: Option<&Path>,
        algorithm: Algorithm,
        export_history: ExportHistory,
        scenario: OptimisationScenario,
    ) -> Result<(), WrapperError> {
        let pywr_problem = PywrProblem::new(model_file, data_path, scenario)?;

        // save the used schema
        Self::save_schema(&pywr_problem.schema, export_history.destination())?;

        // build the optimisation problem
        let objectives = pywr_problem.objectives.clone();
        let variables = pywr_problem.variables();
        let problem = Problem::new(objectives, variables, None, Box::new(pywr_problem))?;

        match algorithm {
            Algorithm::NSGA2(mut options) => {
                if let Some(history) = &mut options.export_history {
                    *history = export_history.clone();
                } else {
                    options.export_history = Some(export_history);
                }
                let mut algorithm = NSGA2::new(problem, options)?;
                algorithm.run()?;
            }
            Algorithm::NSGA3(mut options) => {
                if let Some(history) = &mut options.export_history {
                    *history = export_history.clone();
                } else {
                    options.export_history = Some(export_history);
                }
                let mut algorithm = NSGA3::new(problem, options, false)?;
                algorithm.run()?;
            }
            Algorithm::AdaptiveNSGA3(mut options) => {
                if let Some(history) = &mut options.export_history {
                    *history = export_history.clone();
                } else {
                    options.export_history = Some(export_history);
                }
                let mut algorithm = AdaptiveNSGA3::new(problem, options)?;
                algorithm.run()?;
            }
        };

        Ok(())
    }

    /// Save the used schema in the optimisation to `destination`.
    ///
    /// # Arguments
    ///
    /// * `schema`: The pywr schema.
    /// * `destination`: The destination folder.
    ///
    /// returns: `Result<(), WrapperError>`
    fn save_schema(schema: &PywrModel, destination: &PathBuf) -> Result<(), WrapperError> {
        let data = serde_json::to_string_pretty(schema).map_err(|e| {
            WrapperError::Generic(format!("cannot convert the schema to string because {e}"))
        })?;

        let mut file = destination.to_owned();
        file.push("model_schema.json");

        info!("Saving JSON schema {:?}", file);
        fs::write(file, data).map_err(|e| {
            WrapperError::Generic(format!("cannot export the schema JSON file because {e}",))
        })?;

        Ok(())
    }
}
