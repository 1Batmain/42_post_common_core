use colored::Colorize;
use matrix::{Matrix, Vector};

fn main() {
    println!(
        "{}",
        "ex07 - Matrix/Vector & Matrix/Matrix multiplication"
            .bold()
            .underline()
            .green()
    );

    println!("\n{}", "mul_vec tests".bold().underline().green());

    // Dimension mismatch
    let mat = Matrix::from([1.; 9], 3, 3);
    let vec = Vector::from([3.; 2]);
    println!(
        "3x3 * len2 -> is_err: {}",
        mat.mul_vec(&vec).is_err().to_string().red().bold()
    );

    // Simple 3x3 * vec
    let vec = Vector::from([3.; 3]);
    let res = mat.mul_vec(&vec).unwrap();
    println!("mat:\n{}", mat);
    println!("vec: {}", vec);
    println!("mat * vec = {}\n", res.to_string().bold().red());

    // Non-trivial 2x2 * vec
    let mat = Matrix::from([1., 2., 3., 4.], 2, 2); // [[1,2],[3,4]]
    let vec = Vector::from([5., 6.]);
    let res = mat.mul_vec(&vec).unwrap(); // [17, 39]
    println!("mat:\n{}", mat);
    println!("vec: {}", vec);
    println!("mat * vec = {}\n", res.to_string().bold().red());

    // 1x3 * vec(3)
    let mat = Matrix::from([1., 2., 3.], 3, 1); // [[1,2,3]]
    let vec = Vector::from([4., 5., 6.]);
    let res = mat.mul_vec(&vec).unwrap(); // [32]
    println!("mat:\n{}", mat);
    println!("vec: {}", vec);
    println!("mat * vec = {}\n", res.to_string().bold().red());

    println!("{}", "mul_mat tests".bold().underline().green());

    // Identity check: A * I = A and I * A = A
    let a = Matrix::from([1., 2., 3., 4., 5., 6., 7., 8., 9.], 3, 3);
    let id = Matrix::from([1., 0., 0., 0., 1., 0., 0., 0., 1.], 3, 3);
    println!("A:\n{}", a);
    println!("I:\n{}", id);
    println!(
        "A * I == A : {}",
        (a.mul_mat(&id).unwrap() == a).to_string().bold().red()
    );
    println!(
        "I * A == A : {}",
        (id.mul_mat(&a).unwrap() == a).to_string().bold().red()
    );

    // 2x3 * 3x2 = 2x2
    let a = Matrix::from([1., 2., 3., 4., 5., 6.], 3, 2); // [[1,2,3],[4,5,6]]
    let b = Matrix::from([7., 8., 9., 10., 11., 12.], 2, 3); // [[7,8],[9,10],[11,12]]
    let c = a.mul_mat(&b).unwrap(); // [[58,64],[139,154]]
    println!("\nA (2x3):\n{}", a);
    println!("B (3x2):\n{}", b);
    println!("A * B (2x2):\n{}", c.to_string().bold().red());

    // Dimension mismatch should error
    let a = Matrix::from([1.; 4], 2, 2); // 2x2
    let b = Matrix::from([1.; 6], 2, 3); // 3x2
    println!(
        "\n2x2 * 3x2 -> is_err: {}",
        a.mul_mat(&b).is_err().to_string().red().bold()
    );
}
