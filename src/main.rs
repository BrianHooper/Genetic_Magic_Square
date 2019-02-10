///////////////////////////////////////////////
// An implenetation of a genetic algorithm   //
// for generating magic squares              //
//                                           //
// Brian Hooper                              //
// February 10th, 2019                       //
///////////////////////////////////////////////

extern crate rand;
use rand::{thread_rng, Rng};

// Helper struct for storing a single individual
struct MagicSquare {
    pub values: Vec<u32>,
    pub fitness: f32,
}

// Calculates the magic value for a given width
fn calc_magic_value(width: u32) -> i32 {
    (width as f32 * ((f32::powf(width as f32, 2.0) + 1.0) / 2.0)) as i32
}

// Calculates the total error sum for an individual
fn error(values: &Vec<u32>, width: u32, magic_value: i32) -> i32 {
    let mut row_sum: i32 = 0;
    let mut col_sum: i32 = 0;

    for row in (0..values.len()).step_by(width as usize) {
        let mut sum: i32 = 0;
        for col in 0..width {
            sum = sum + values[row + col as usize] as i32;
        }
        row_sum = row_sum + (magic_value - sum).abs();
    }

    for col in 0..width {
        let mut sum: i32 = 0;
        for row in (0..values.len()).step_by(width as usize) {
            sum = sum + values[row + col as usize] as i32;
        }
        col_sum = col_sum + (magic_value - sum).abs();
    }

    let mut i = 0;
    let mut i_sum: i32 = 0;
    let mut j = values.len() - (width as usize);
    let mut j_sum: i32 = 0;

    while i < values.len() && j > 0 {
        i_sum = i_sum + (values[i] as i32);
        i = i + (width as usize) + 1;
        j_sum = j_sum + (values[j] as i32);
        j = j - ((width as usize) - 1);
    }

    let total_error = col_sum + row_sum + (magic_value - i_sum).abs() + (magic_value - j_sum).abs();

    total_error
}

// Calculates the fitness of an individual based on the total error sum
fn fitness(values: &Vec<u32>, width: u32, magic_value: i32) -> f32 {
    let total_error = error(values, width, magic_value);
    let fit: f32;
    if total_error == 0 {
        fit = 1.0;
    } else {
        fit = 1.0 / (total_error as f32);
    }
    fit
}

// Caclualtes the total fitness for a population
fn calculate_total_fitness(individuals: &Vec<MagicSquare>) -> f32 {
    let mut total_fitness = 0.0;
    for individual in individuals.iter() {
        total_fitness = total_fitness + &individual.fitness;
    }
    total_fitness
}

// Sorts a population in descending order based on the fitness of each individual
fn sort(individuals: &mut Vec<MagicSquare>) {
    individuals.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
}

// Mutates an individual, based on a mutation rate
pub fn mutate(values: Vec<u32>, mutation_rate: f32) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    if rng.gen_range(0.0, 1.0) < mutation_rate {
        let mut mutated_values = values.clone();
        let index_a = rng.gen_range(0, values.len());
        let index_b = rng.gen_range(0, values.len());

        let temp = mutated_values[index_a];
        mutated_values[index_a] = mutated_values[index_b];
        mutated_values[index_b] = temp;
        mutated_values
    } else {
        values
    }
}
// Generates a population
fn create_population(width: u32, size: usize, magic_value: i32) -> Vec<MagicSquare> {
    let length = u32::pow(width, 2);
    let mut individuals: Vec<MagicSquare> = Vec::new();
    let mut rng = thread_rng();

    for _x in 0..size {
        let mut vec: Vec<u32> = (1..=length).collect();
        rng.shuffle(&mut vec);
        let fitness = fitness(&vec, width, magic_value);
        individuals.push(MagicSquare {
            values: vec,
            fitness,
        });
    }

    individuals
}

// Selects a parent using roulette-wheel selection
fn select(population: &Vec<MagicSquare>, total_fitness: f32) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    let mut r = rng.gen_range(0.0001, total_fitness);

    let mut i = 0;
    while r > 0.0 {
        r = r - population[i].fitness;
        i = i + 1;
    }

    population[i - 1].values.clone()
}

// Performs a single-index crossover using inversions
fn crossover(parent_1: &Vec<u32>, parent_2: &Vec<u32>) -> (Vec<u32>, Vec<u32>) {
    let parent_1_inversion: Vec<u32> = create_inversion(&parent_1);
    let parent_2_inversion: Vec<u32> = create_inversion(&parent_2);

    let mut rng = thread_rng();
    let cr_index = rng.gen_range(0, parent_1_inversion.len());

    let mut child_1_inversion: Vec<u32> = Vec::new();
    let mut child_2_inversion: Vec<u32> = Vec::new();

    for i in 0..cr_index {
        child_1_inversion.push(parent_1_inversion[i]);
        child_2_inversion.push(parent_2_inversion[i]);
    }

    for i in cr_index..parent_1.len() {
        child_1_inversion.push(parent_2_inversion[i]);
        child_2_inversion.push(parent_1_inversion[i]);
    }

    let child_1 = transform_inversion(&child_1_inversion);
    let child_2 = transform_inversion(&child_2_inversion);

    (child_1, child_2)
}

// Converts an individual to its inversion form
fn create_inversion(values: &Vec<u32>) -> Vec<u32> {
    let mut inversion: Vec<u32> = values.clone();

    for i in 1..=values.len() as u32 {
        let mut index = 0;
        let mut num_greater = 0;

        while values[index] != i {
            if values[index] > i {
                num_greater = num_greater + 1;
            }
            index = index + 1;
        }
        inversion[(i - 1) as usize] = num_greater;
    }

    inversion
}

// Transforms an inversion back into an individual
fn transform_inversion(inversion: &Vec<u32>) -> Vec<u32> {
    let mut intermediate: Vec<u32> = vec![0; inversion.len()];

    for i in (0..=(inversion.len() - 1)).rev() {
        intermediate[i] = inversion[i];
        for j in (i + 1)..inversion.len() {
            if intermediate[j] >= intermediate[i] {
                intermediate[j] = intermediate[j] + 1;
            }
        }
    }

    let mut original: Vec<u32> = inversion.clone();

    for i in 0..intermediate.len() {
        original[intermediate[i] as usize] = (i + 1) as u32;
    }

    original
}

// Combines the best individuals of each population into a single population
fn reduce(population_1: Vec<MagicSquare>, population_2: Vec<MagicSquare>) -> Vec<MagicSquare> {
    let mut combined_population: Vec<MagicSquare> = Vec::new();

    let mut i: usize = 0;
    let mut j: usize = 0;

    while combined_population.len() < population_1.len() {
        if population_1[i].fitness > population_2[j].fitness {
            let individual = MagicSquare {
                fitness: population_1[i].fitness,
                values: population_1[i].values.clone(),
            };
            combined_population.push(individual);
            i = i + 1;
        } else {
            let individual = MagicSquare {
                fitness: population_2[j].fitness,
                values: population_2[j].values.clone(),
            };
            combined_population.push(individual);
            j = j + 1;
        }
    }

    combined_population
}

// Runs a genetic algorithm and returns the best individual in the population
fn genetic_algorithm(
    mut population_1: Vec<MagicSquare>,
    width: u32,
    magic_value: i32,
    iterations: i32,
    mutation_rate: f32,
) -> Vec<u32> {
    sort(&mut population_1);
    let mut iteration = 0;
    while iteration < iterations {
        iteration = iteration + 1;
        // Early stopping
        if error(&population_1[0].values, width, magic_value) == 0 {
            break;
        }

        let total_fitness = calculate_total_fitness(&population_1);
        let mut population_2: Vec<MagicSquare> = Vec::new();

        for _x in (0..population_1.len()).step_by(2) {
            let parent_1: Vec<u32> = select(&population_1, total_fitness);
            let parent_2: Vec<u32> = select(&population_1, total_fitness);

            let (child_1, child_2) = crossover(&parent_1, &parent_2);
            let child_1 = mutate(child_1, mutation_rate);
            population_2.push(MagicSquare {
                fitness: fitness(&child_1, width, magic_value),
                values: child_1,
            });

            let child_2 = mutate(child_2, mutation_rate);
            population_2.push(MagicSquare {
                fitness: fitness(&child_2, width, magic_value),
                values: child_2,
            });
        }
        sort(&mut population_2);
        population_1 = reduce(population_1, population_2);
    }
    println!("Total iterations: {}", iteration);
    let result: Vec<u32> = population_1[0].values.clone();
    result
}

// Outputs the magic square to the console
fn print_square(square: &Vec<u32>, width: u32) {
    for i in (0..square.len()).step_by(width as usize) {
        for j in 0..width {
            print!("{}\t", square[i + j as usize]);
        }
        println!();
    }
}

fn main() {
    let width: u32 = 4; // width of the magic square
    let size: usize = 100; // number of individuals in the population
    let iterations: i32 = 500; // maximum number of iterations
    let mutation_rate: f32 = 0.01; // mutation rate

    if width < 3 {
        eprintln!("Error, width should be greater than 2");
        std::process::exit(1);
    }

    if size < 2 {
        eprintln!("Error, population must have at least 2 individuals");
        std::process::exit(1);
    }

    if iterations < 1 {
        eprintln!("Error, must have at least one iteration");
        std::process::exit(1);
    }

    let magic_value: i32 = calc_magic_value(width);
    println!(
        "Finding magic square of width {} with magic value of {}:",
        width, magic_value
    );

    // Create initial population
    let individuals: Vec<MagicSquare> = create_population(width, size, magic_value);

    // Run genetic algorithm
    let result = genetic_algorithm(individuals, width, magic_value, iterations, mutation_rate);

    // Output results
    println!("Error: {}", error(&result, width, magic_value));
    print_square(&result, width);
}
