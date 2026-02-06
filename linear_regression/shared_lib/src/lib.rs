use serde::{Serialize, Deserialize};

#[derive(Default, Debug, Deserialize, Serialize)]
pub struct Model
{
    pub theta0: f64,
    pub theta1: f64,
    pub mean: f64,
    pub standard_deviation: f64,
}


pub fn normalize(km: u64, model: &Model) -> f64
{
    (km as f64 - model.mean) / model.standard_deviation
}

pub fn predict(km: u64, model: &Model) -> f64
{
    let km = normalize(km, model);
    model.theta0 + km * model.theta1
}

