use log::error;
use optirustic::core::OError;
use thiserror::Error;

/// Errors raised by the library
#[derive(Error, Debug)]
pub enum WrapperError {
    #[error("The following error occurred: {0}")]
    Generic(String),
    #[error("The following optirustic error occurred: {0}")]
    Optirustic(#[from] OError),
    #[error("The following pywr error occurred: {0}")]
    Pywr(String),
    #[error("The parameter ({0}) type is not supported")]
    NotSupportedParameterType(String),
    #[error("Cannot determine the number of points in the RBF parameter '{0}' because the property 'points' is empty")]
    EmptyRBFProfile(String),
    #[error("Cannot setup the optimiser because the model does not contain any parameter")]
    NoModelParameters,
    #[error("The model does not contain any variable parameters. Make sure the parameters have been properly set up to use")]
    NoVariableParameters,
    #[error("The bounds for parameter {0} are invalid because: {1}")]
    InvalidParameterValueBounds(String, String),
    #[error("The bounds for constraint {0} are invalid because: {1}")]
    InvalidConstraintBounds(String, String),
    #[error("The recorder named '{0}' does not exist")]
    MissingRecorder(String),
}
