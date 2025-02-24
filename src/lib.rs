mod error;
mod problem;
mod scenario;

use optirustic::algorithms::{AdaptiveNSGA3, Algorithm as OAlgorithm, NSGA2, NSGA3};
use std::path::Path;

use crate::error::WrapperError;
use crate::problem::PywrProblem;
use crate::scenario::OptimisationScenario;
use optirustic::algorithms::{NSGA2Arg, NSGA3Arg};
use optirustic::core::Problem;

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
    /// * `scenario`: The optimisation scenario used to define the objectives and constraints.
    ///
    /// returns: `Result<(), WrapperError>`
    pub fn run(
        model_file: &Path,
        data_path: Option<&Path>,
        algorithm: Algorithm,
        scenario: OptimisationScenario,
    ) -> Result<(), WrapperError> {
        let pywr_problem = PywrProblem::new(model_file, data_path, scenario)?;

        let objectives = pywr_problem.objectives.clone();
        let variables = pywr_problem.variables();
        let problem = Problem::new(objectives, variables, None, Box::new(pywr_problem))?;

        match algorithm {
            Algorithm::NSGA2(options) => {
                let mut algorithm = NSGA2::new(problem, options)?;
                algorithm.run()?;
            }
            Algorithm::NSGA3(options) => {
                let mut algorithm = NSGA3::new(problem, options, false)?;
                algorithm.run()?;
            }
            Algorithm::AdaptiveNSGA3(options) => {
                let mut algorithm = AdaptiveNSGA3::new(problem, options)?;
                algorithm.run()?;
            }
        };

        Ok(())
    }
}
