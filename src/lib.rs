mod error;
mod problem;
mod scenario;

use crate::error::WrapperError;
use crate::problem::PywrProblem;
use crate::scenario::OptimisationScenario;
use log::info;
use optirustic::algorithms::{
    AdaptiveNSGA3, Algorithm as OAlgorithm, ExportHistory, PyAlgorithm, NSGA2, NSGA3,
};
use optirustic::core::Problem;
use pyo3::prelude::*;
use pywr_schema::PywrModel;
use std::fs;
use std::path::PathBuf;

/// The optirustic wrapper to solve optimisation problems using pywr.
#[pyclass]
pub struct PywrOptirustic;

#[pymethods]
impl PywrOptirustic {
    #[new]
    fn new() -> Self {
        PywrOptirustic {}
    }

    /// Solve the problem with the chosen algorithm.
    ///
    /// # Arguments
    ///
    /// * `model_file`: The path to the model JSON file.
    /// * `algorithm`: The algorithm and its option to use to solve the optimisation problem.
    /// * `export_history`: The data used to configure the result export for the generations. The
    ///    option in  'algorithm.export_history', if provided, is overwritten by this argument.
    /// * `scenario_file`: The optimisation scenario file to use to define the objectives and
    ///    constraints.
    /// * `data_path`: The optional path where the model data is stored.
    ///
    /// returns: `PyResult<()>`
    #[staticmethod]
    #[pyo3(signature = (model_file, algorithm, export_history, scenario_file, data_path=None))]
    fn run(
        model_file: PathBuf,
        algorithm: PyAlgorithm,
        export_history: ExportHistory,
        scenario_file: PathBuf,
        data_path: Option<PathBuf>,
    ) -> PyResult<()> {
        let scenario = OptimisationScenario::load_from_file(&scenario_file)?;

        let pywr_problem = PywrProblem::new(model_file, data_path, scenario)?;

        // save the used schema
        Self::save_schema(&pywr_problem.schema, export_history.destination())?;

        // build the optimisation problem
        let objectives = pywr_problem.objectives.clone();
        let variables = pywr_problem.variables();
        let problem = Problem::new(objectives, variables, None, Box::new(pywr_problem))?;

        match algorithm {
            PyAlgorithm::nsga2 { mut options } => {
                if let Some(history) = &mut options.export_history {
                    *history = export_history.clone();
                } else {
                    options.export_history = Some(export_history);
                }
                let mut algorithm = NSGA2::new(problem, options)?;
                algorithm.run()?
            }
            PyAlgorithm::nsga3 { mut options } => {
                if let Some(history) = &mut options.export_history {
                    *history = export_history.clone();
                } else {
                    options.export_history = Some(export_history);
                }
                let mut algorithm = NSGA3::new(problem, options, false)?;
                algorithm.run()?
            }
            PyAlgorithm::adaptive_nsga3 { mut options } => {
                if let Some(history) = &mut options.export_history {
                    *history = export_history.clone();
                } else {
                    options.export_history = Some(export_history);
                }
                let mut algorithm = AdaptiveNSGA3::new(problem, options)?;
                algorithm.run()?
            }
        };

        Ok(())
    }
}

impl PywrOptirustic {
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

#[pymodule]
fn pywr_optirustic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OptimisationScenario>()?;
    m.add_class::<PywrOptirustic>()?;

    Ok(())
}
