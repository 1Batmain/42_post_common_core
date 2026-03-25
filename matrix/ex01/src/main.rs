use colored::Colorize;
use matrix::{Vector, linear_combination};

fn main() {
    let e1 = Vector::from([1., 0., 0.]);
    let e2 = Vector::from([0., 1., 0.]);
    let e3 = Vector::from([0., 0., 1.]);
    let v1 = Vector::from([1., 2., 3.]);
    let v2 = Vector::from([0., 10., -100.]);
    println!("{:^21}|{:^21}", "expected", "result".bold());
    println!(
        "{:^21}|{:^21}",
        "[10., -2., 0.5]",
        linear_combination([e1, e2, e3], [10., -2., 0.5])
            .unwrap()
            .to_string()
            .bold()
    );
    //
    println!(
        "{:^21}|{:^21}",
        "[10., 0., 230]",
        linear_combination([v1, v2], [10., -2.])
            .unwrap()
            .to_string()
            .bold()
    );
    // Should panic
    // let e1 = Vector::from([1., 0., 0., 4.]); // Vectors of different shapes
    // let e2 = Vector::from([0., 1., 0.]);
    // let e3 = Vector::from([0., 0., 1.]);
    // linear_combination([e1, e2, e3], [10., -2., 0.5]);

    // let e1 = Vector::from([1., 0., 0.]); // Scalar list doesnt match number of vectors
    // let e2 = Vector::from([0., 1., 0.]);
    // let e3 = Vector::from([0., 0., 1.]);
    // linear_combination([e1, e2, e3], [10., -2., 0.5, 3.]);
}
