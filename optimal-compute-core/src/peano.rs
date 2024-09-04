pub use self::{add::*, max::*};

#[derive(Clone, Copy, Debug)]
pub struct Zero;

#[derive(Clone, Copy, Debug)]
pub struct Suc<A>(A);

pub type One = Suc<Zero>;
pub type Two = Suc<One>;

mod max {
    use super::*;

    pub trait Max<B> {
        type Max;
    }

    impl Max<Zero> for Zero {
        type Max = Zero;
    }

    impl<A> Max<Zero> for Suc<A> {
        type Max = Suc<A>;
    }

    impl<B> Max<Suc<B>> for Zero {
        type Max = Suc<B>;
    }

    impl<A, B> Max<Suc<B>> for Suc<A>
    where
        A: Max<B>,
    {
        type Max = Suc<<A as Max<B>>::Max>;
    }
}

mod add {
    use super::*;

    pub trait Add<B> {
        type Add;
    }

    impl Add<Zero> for Zero {
        type Add = Zero;
    }

    impl<A> Add<Zero> for Suc<A> {
        type Add = Suc<A>;
    }

    impl<B> Add<Suc<B>> for Zero {
        type Add = Suc<B>;
    }

    impl<A, B> Add<Suc<B>> for Suc<A>
    where
        A: Add<B>,
    {
        type Add = Suc<Suc<<A as Add<B>>::Add>>;
    }
}
