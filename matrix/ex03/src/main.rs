use colored::Colorize;
use ex03::{Vector, dot};

fn main() {
    println!(
        "{}",
        "Testing orthogonal vectors".bold().underline().green()
    );
    let v1 = Vector::from([0, 1]);
    let v2 = Vector::from([1, 0]);
    println!("v1->{}", v1);
    println!("v2->{}", v2);
    println!("v1.v2-> {}", dot(&v1, &v2).to_string().bold().green());
    let v1 = Vector::from([1, 1]);
    let v2 = Vector::from([-1, 1]);
    println!("v1->{}", v1);
    println!("v2->{}", v2);
    println!("v1.v2-> {}\n", dot(&v1, &v2).to_string().bold().green());

    println!("{}", "Tests from subjects".bold().underline().green());
    let v1 = Vector::from([1, 1]);
    let v2 = Vector::from([1, 1]);
    println!("v1->{}", v1);
    println!("v2->{}", v2);
    println!("v1.v2-> {}", dot(&v1, &v2).to_string().bold().green());
    let v1 = Vector::from([-1, 6]);
    let v2 = Vector::from([3, 2]);
    println!("v1->{}", v1);
    println!("v2->{}", v2);
    println!("v1.v2-> {}\n", dot(&v1, &v2).to_string().bold().green());
}
