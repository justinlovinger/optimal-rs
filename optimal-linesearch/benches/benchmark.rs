use std::{hint::black_box, vec::Vec};

use fixed_step_size::FixedStepSize;
use optimal_compute_core::{
    arg1, named_args,
    enumerate::Enumerate,
    math::{Add, Div, Mul, Pow},
    peano::{One, Zero},
    sum::Sum,
    val,
    zip::Zip,
    Arg, Computation, ComputationFn, Run, Val,
};
use optimal_linesearch::backtracking_line_search::{
    BacktrackingLineSearchBuilder, BacktrackingLineSearchComputation,
    BacktrackingLineSearchStoppingCriteria, BfgsInitializer, StepDirection,
};
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main, IntoBenchmarks};

pub fn linesearch_benchmarks() -> impl IntoBenchmarks {
    let len = 100000;

    // BFGS requires quadratic time and memory,
    // relative to the length of points.
    let bfgs_len = 400;

    [
        benchmark_fn(
            format!("fixed_step_size steepest skewed_sphere {len}"),
            move |b| {
                b.iter(move || {
                    run_fixed_step_size(
                        black_box(skewed_sphere_d()),
                        black_box(skewed_sphere_initial_point(len)),
                    )
                })
            },
        ),
        benchmark_fn(
            format!("backtracking_line_search steepest skewed_sphere {len}"),
            move |b| {
                b.iter(move || {
                    run_backtracking_line_search(
                        StepDirection::Steepest,
                        black_box(skewed_sphere()),
                        black_box(skewed_sphere_d()),
                        black_box(skewed_sphere_initial_point(len)),
                    )
                })
            },
        ),
        benchmark_fn(
            format!("backtracking_line_search bfgs skewed_sphere {bfgs_len}"),
            move |b| {
                b.iter(move || {
                    run_backtracking_line_search(
                        StepDirection::Bfgs {
                            initializer: BfgsInitializer::Gamma,
                        },
                        black_box(skewed_sphere()),
                        black_box(skewed_sphere_d()),
                        black_box(skewed_sphere_initial_point(bfgs_len)),
                    )
                })
            },
        ),
    ]
}

fn skewed_sphere_initial_point(len: usize) -> Vec<f64> {
    (1..(len + 1)).map(|x| x as f64).collect()
}

pub fn run_fixed_step_size<FD>(obj_func_d: FD, initial_point: Vec<f64>) -> Vec<f64>
where
    FD: Clone + ComputationFn<Dim = One, Item = f64>,
    FixedStepSize<FD>: Run<Output = Vec<f64>>,
{
    fixed_step_size::fixed_step_size(obj_func_d, initial_point).run(named_args![])
}

pub fn run_backtracking_line_search<F, FD>(
    step_direction: StepDirection,
    obj_func: F,
    obj_func_d: FD,
    initial_point: Vec<f64>,
) -> Vec<f64>
where
    F: Clone + ComputationFn<Dim = Zero, Item = f64>,
    FD: Clone + ComputationFn<Dim = One, Item = f64>,
    BacktrackingLineSearchComputation<f64, F, Zip<F, FD>>: Run<Output = Vec<f64>>,
{
    BacktrackingLineSearchBuilder::default()
        .direction(step_direction)
        .stopping_criteria(BacktrackingLineSearchStoppingCriteria::Iteration(100))
        .for_(initial_point.len(), obj_func, obj_func_d)
        .with_point(initial_point)
        .argmin()
}

fn skewed_sphere() -> SkewedSphere {
    arg1!("point", f64)
        .enumerate(arg1!("x", f64).pow(val!(1.0) + (arg1!("i", f64) + val!(1.0)) / val!(100.0)))
        .sum()
}

type SkewedSphere = Sum<
    Enumerate<
        Arg<One, f64>,
        Pow<
            Arg<One, f64>,
            Add<Val<Zero, f64>, Div<Add<Arg<One, f64>, Val<Zero, f64>>, Val<Zero, f64>>>,
        >,
    >,
>;

fn skewed_sphere_d() -> SkewedSphereD {
    arg1!("point", f64)
        .enumerate((val!(1.0) + (arg1!("i", f64) + val!(1.0)) / val!(100.0)) * arg1!("x", f64))
}

type SkewedSphereD = Enumerate<
    Arg<One, f64>,
    Mul<
        Add<Val<Zero, f64>, Div<Add<Arg<One, f64>, Val<Zero, f64>>, Val<Zero, f64>>>,
        Arg<One, f64>,
    >,
>;

tango_benchmarks!(linesearch_benchmarks());
tango_main!();

mod fixed_step_size {
    use optimal_compute_core::{
        arg, arg1,
        cmp::Lt,
        control_flow::LoopWhile,
        math::{Add, Mul, Neg},
        peano::{One, Zero},
        val, val1,
        zip::{Snd, Zip},
        Arg, Computation, ComputationFn, Val,
    };
    use optimal_linesearch::{descend, step_direction::steepest_descent, StepSize};

    pub fn fixed_step_size<FD>(obj_func_d: FD, initial_point: Vec<f64>) -> FixedStepSize<FD>
    where
        FD: Clone + ComputationFn<Dim = One, Item = f64>,
    {
        val!(0)
            .zip(val1!(initial_point))
            .loop_while(
                ("i", "point"),
                (arg!("i", usize) + val!(1)).zip(descend(
                    val!(StepSize::new(0.5).unwrap()),
                    steepest_descent(obj_func_d),
                    arg1!("point", f64),
                )),
                arg!("i", usize).lt(val!(2000)),
            )
            .snd()
    }

    pub type FixedStepSize<FD> = Snd<
        LoopWhile<
            Zip<Val<Zero, usize>, Val<One, Vec<f64>>>,
            (&'static str, &'static str),
            Zip<
                Add<Arg<Zero, usize>, Val<Zero, usize>>,
                Add<Arg<One, f64>, Mul<Val<Zero, StepSize<f64>>, Neg<FD>>>,
            >,
            Lt<Arg<Zero, usize>, Val<Zero, usize>>,
        >,
    >;
}
