use colored::Colorize;
use matrix::{Vector, cross_product};

fn main() {
    println!("{}", "ex06 - cross product".bold().underline().green());

    let a = Vector::from([1., 0., 0.]);
    let b = Vector::from([0., 1., 0.]);
    let expected = Vector::from([0., 0., 1.]);

    let res = cross_product(&a, &b).unwrap();

    println!("a       -> {}", a);
    println!("b       -> {}", b);
    println!("expected-> {}", expected.to_string().yellow());
    println!("result  -> {}", res.to_string().bold().red());

    assert_eq!(res, expected);

    println!();
    println!(
        "{}",
        "error case (non-3D vectors)".bold().underline().green()
    );

    let a_bad = Vector::from([1., 0.]);
    let b_bad = Vector::from([0., 1.]);

    let err = cross_product(&a_bad, &b_bad);
    println!("a_bad -> {}", a_bad);
    println!("b_bad -> {}", b_bad);
    println!("cross_product(a_bad, b_bad) is_err -> {}", err.is_err());

    assert!(err.is_err());

    println!();
    println!("{}", "ex06 completed successfully".bold().green());
}
