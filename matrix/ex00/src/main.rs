use colored::Colorize;
use ex00::{Matrix, Vector};

fn main() {
    println!(
        "{}",
        "\nTesting Vector creation & display"
            .to_string()
            .bold()
            .underline()
            .green()
    );
    let vec = Vector::from([1, 2, 3]);
    println!("Vector::from([1., 2., 3.]) -> {}", vec);
    let vec = Vector::from([1., 2., 3.]);
    println!("Vector::from([1, 2, 3]) -> {}", vec);
    let vec = Vector::from([0; 10]);
    println!("Vector::from([0; 10]) -> {}", vec);
    println!(
        "{}",
        "\nTesting Matrix creation & display"
            .to_string()
            .bold()
            .underline()
            .green()
    );
    let mat = Matrix::from([1, 2, 3], 1, 3);
    println!("Matrix::from([1., 2., 3.], 1, 3) \n{}", mat);
    let mat = Matrix::from([1., 2., 3.], 3, 1);
    println!("Matrix::from([1, 2, 3], 3, 1) \n{}", mat);
    let mat = Matrix::from([0; 10], 2, 5);
    println!("Matrix::from([0; 10], 2, 5) \n{}", mat);
    // Should Panic
    // let mat = Matrix::from([0; 10], 20, 5);
    // println!("Matrix::from([0; 10], 2, 5) \n{}", mat);
    println!(
        "{}",
        "\nTesting add function"
            .to_string()
            .bold()
            .underline()
            .green()
    );
    let a = Vector::from([1; 10]);
    let b = Vector::from([2; 10]);
    let mut r = a.clone();
    r.add(&b);
    println!("a->{}\nb->{}\nr->{}", a, b, r.to_string().bold().red());
    let a = Matrix::from([1; 10], 2, 5);
    let b = Matrix::from([2; 10], 2, 5);
    let mut r = a.clone();
    r.add(&b);
    println!("a->{}\nb->{}\nr->{}", a, b, r.to_string().bold().red());
    // Should panic (Not same size)
    // let a = Vector::from([1; 10]);
    // let b = Vector::from([2; 15]);
    // let mut r = a.clone();
    // r.add(&b);
    // println!("a->{}\nb->{}\nr->{}", a, b, r.to_string().bold().red());
    // let a = Matrix::from([1; 10], 2, 5);
    // let b = Matrix::from([2; 10], 5, 2);
    // let mut r = a.clone();
    // r.add(&b);
    // println!("a->{}\nb->{}\nr->{}", a, b, r.to_string().bold().red());
    println!(
        "{}",
        "\nTesting sub function"
            .to_string()
            .bold()
            .underline()
            .green()
    );
    let a = Vector::from([1; 10]);
    let b = Vector::from([2; 10]);
    let mut r = a.clone();
    r.sub(&b);
    println!("a->{}\nb->{}\nr->{}", a, b, r.to_string().bold().red());
    let a = Matrix::from([1; 10], 2, 5);
    let b = Matrix::from([2; 10], 2, 5);
    let mut r = a.clone();
    r.sub(&b);
    println!("a->{}\nb->{}\nr->{}", a, b, r.to_string().bold().red());
    // Should panic (Not same size)
    // let a = Vector::from([1; 10]);
    // let b = Vector::from([2; 15]);
    // let mut r = a.clone();
    // r.sub(&b);
    // println!("a->{}\nb->{}\nr->{}", a, b, r.to_string().bold().red());
    // let a = Matrix::from([1; 10], 2, 5);
    // let b = Matrix::from([2; 10], 5, 2);
    // let mut r = a.clone();
    // r.sub(&b);
    // println!("a->{}\nb->{}\nr->{}", a, b, r.to_string().bold().red());
    println!(
        "{}",
        "\nTesting scale function"
            .to_string()
            .bold()
            .underline()
            .green()
    );
    let scaler: usize = 42;
    let mut a = Vector::from([1; 10]);
    println!("{:^7}->{}", "a", a);
    a.scl(scaler);
    println!("{:^7}->{}", format!("{} * a", scaler), a);
    let mut a = Matrix::from([1; 9], 3, 3);
    println!("{:^7}->{}", "a", a);
    a.scl(scaler);
    println!("{:^7}->{}", format!("{} * a", scaler), a);
}
