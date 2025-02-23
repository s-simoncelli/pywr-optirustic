mod error;
mod problem;

use optirustic::algorithms::{
    AdaptiveNSGA3, Algorithm as OAlgorithm, MaxGenerationValue, StoppingConditionType, NSGA2, NSGA3,
};
use std::error::Error;
use std::path::{Path, PathBuf};

use crate::error::WrapperError;
use crate::problem::PywrProblem;
use log::{info, LevelFilter};
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
    ///
    /// returns: `Box<PywrOptirustic>`
    pub fn run(
        model_file: &Path,
        data_path: Option<&Path>,
        algorithm: Algorithm,
    ) -> Result<(), WrapperError> {
        let pywr_problem = PywrProblem::new(model_file, data_path)?;

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

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .init();

    info!("Loading JSON file");
    let model_file = PathBuf::from("./data/timeseries.json");
    let problem = PywrProblem::new(&model_file, None)?;

    println!("{:?}", problem.variables());

    let algorithm = Algorithm::NSGA2(NSGA2Arg {
        number_of_individuals: 100,
        stopping_condition: StoppingConditionType::MaxGeneration(MaxGenerationValue(250)),
        crossover_operator_options: None,
        mutation_operator_options: None,
        parallel: Some(false),
        export_history: None,
        resume_from_file: None,
        seed: Some(10),
    });
    PywrOptirustic::run(&model_file, None, algorithm)?;

    Ok(())
}
