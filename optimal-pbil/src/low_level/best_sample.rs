use computation_types::{
    arg, arg1,
    black_box::BlackBox,
    cmp::Lt,
    control_flow::{If, LoopWhile, Then},
    math::Add,
    peano::{One, Zero},
    rand::SeededRand,
    run::Value,
    val,
    zip::{Zip, Zip3, Zip4},
    Arg, Computation, ComputationFn, Val,
};
use rand::{distributions::Bernoulli, Rng};

use crate::types::{NumSamples, Probability};

/// See [`best_sample`].
pub type BestSample<N, F, P, R> = Then<
    LoopWhile<
        Zip<
            Zip<Val<Zero, usize>, N>,
            Then<
                Zip<BlackBox<P, fn(Vec<Probability>) -> Value<Vec<Bernoulli>>, One, Bernoulli>, R>,
                (&'static str, &'static str),
                Zip<
                    Arg<One, Bernoulli>,
                    Then<
                        SeededRand<Arg<Zero, <R as Computation>::Item>, Arg<One, Bernoulli>, bool>,
                        (&'static str, &'static str),
                        Zip<Arg<Zero, <R as Computation>::Item>, Zip<Arg<One, bool>, F>>,
                    >,
                >,
            >,
        >,
        (
            (&'static str, &'static str),
            (&'static str, (&'static str, (&'static str, &'static str))),
        ),
        Zip<
            Zip<Add<Arg<Zero, usize>, Val<Zero, usize>>, Arg<Zero, NumSamples>>,
            Zip<
                Arg<One, Bernoulli>,
                Then<
                    Zip3<
                        Arg<One, bool>,
                        Arg<Zero, <F as Computation>::Item>,
                        SeededRand<Arg<Zero, <R as Computation>::Item>, Arg<One, Bernoulli>, bool>,
                    >,
                    (&'static str, &'static str, (&'static str, &'static str)),
                    Zip<
                        Arg<Zero, <R as Computation>::Item>,
                        Then<
                            Zip4<
                                Arg<One, bool>,
                                Arg<Zero, <F as Computation>::Item>,
                                Arg<One, bool>,
                                F,
                            >,
                            (&'static str, &'static str, &'static str, &'static str),
                            If<
                                Zip4<
                                    Arg<One, bool>,
                                    Arg<Zero, <F as Computation>::Item>,
                                    Arg<One, bool>,
                                    Arg<Zero, <F as Computation>::Item>,
                                >,
                                (&'static str, &'static str, &'static str, &'static str),
                                Lt<
                                    Arg<Zero, <F as Computation>::Item>,
                                    Arg<Zero, <F as Computation>::Item>,
                                >,
                                Zip<Arg<One, bool>, Arg<Zero, <F as Computation>::Item>>,
                                Zip<Arg<One, bool>, Arg<Zero, <F as Computation>::Item>>,
                            >,
                        >,
                    >,
                >,
            >,
        >,
        Lt<Arg<Zero, usize>, Arg<Zero, NumSamples>>,
    >,
    (
        (&'static str, &'static str),
        (&'static str, (&'static str, (&'static str, &'static str))),
    ),
    Zip<Arg<Zero, <R as Computation>::Item>, Arg<One, bool>>,
>;

/// Return the sample that minimizes the objective function
/// among `num_samples` samples
/// sampled from `probabilities`.
pub fn best_sample<N, F, P, R>(
    num_samples: N,
    obj_func: F,
    probabilities: P,
    rng: R,
) -> BestSample<N, F, P, R>
where
    N: Computation<Dim = Zero, Item = NumSamples>,
    F: Clone + ComputationFn<Dim = Zero>,
    F::Item: PartialOrd,
    P: Computation<Dim = One, Item = Probability>,
    R: Computation<Dim = Zero>,
    R::Item: Rng,
{
    Zip(
        Zip(val!(1_usize), num_samples),
        Zip(
            probabilities.black_box::<_, One, Bernoulli>(
                bernoullis_from_probabilities as fn(Vec<Probability>) -> Value<Vec<Bernoulli>>,
            ),
            rng,
        )
        .then(
            ("distributions", "rng"),
            Zip(
                arg1!("distributions", Bernoulli),
                SeededRand::<_, _, bool>::new(
                    arg!("rng", R::Item),
                    arg1!("distributions", Bernoulli),
                )
                .then(
                    ("rng", "sample"),
                    Zip(
                        arg!("rng", R::Item),
                        Zip(arg1!("sample", bool), obj_func.clone()),
                    ),
                ),
            ),
        ),
    )
    .loop_while(
        (
            ("i", "num_samples"),
            ("distributions", ("rng", ("best_sample", "best_value"))),
        ),
        Zip(
            Zip(
                arg!("i", usize) + val!(1_usize),
                arg!("num_samples", NumSamples),
            ),
            Zip(
                arg1!("distributions", Bernoulli),
                Zip3(
                    arg1!("best_sample", bool),
                    arg!("best_value", F::Item),
                    SeededRand::<_, _, bool>::new(
                        arg!("rng", R::Item),
                        arg1!("distributions", Bernoulli),
                    ),
                )
                .then(
                    ("best_sample", "best_value", ("rng", "sample")),
                    Zip(
                        arg!("rng", R::Item),
                        Zip4(
                            arg1!("best_sample", bool),
                            arg!("best_value", F::Item),
                            arg1!("sample", bool),
                            obj_func,
                        )
                        .then(
                            ("best_sample", "best_value", "sample", "value"),
                            Zip4(
                                arg1!("best_sample", bool),
                                arg!("best_value", F::Item),
                                arg1!("sample", bool),
                                arg!("value", F::Item),
                            )
                            .if_(
                                ("best_sample", "best_value", "sample", "value"),
                                arg!("value", F::Item).lt(arg!("best_value", F::Item)),
                                Zip(arg1!("sample", bool), arg!("value", F::Item)),
                                Zip(arg1!("best_sample", bool), arg!("best_value", F::Item)),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        arg!("i", usize).lt(arg!("num_samples", NumSamples)),
    )
    .then(
        (
            ("i", "num_samples"),
            ("distributions", ("rng", ("best_sample", "best_value"))),
        ),
        Zip(arg!("rng", R::Item), arg1!("best_sample", bool)),
    )
}

fn bernoullis_from_probabilities(probabilities: Vec<Probability>) -> Value<Vec<Bernoulli>> {
    Value(
        probabilities
            .into_iter()
            .map(|p| Bernoulli::new(f64::from(p)).expect("Probability should be valid"))
            .collect::<Vec<_>>(),
    )
}
