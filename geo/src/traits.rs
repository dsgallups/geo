use crate::geometry::{Coordinate, Point};

use crate::CoordNum;

pub trait Coord: PartialEq {
    type Scalar: CoordNum;

    fn x(&self) -> Self::Scalar;
    fn y(&self) -> Self::Scalar;
    fn x_mut(&mut self) -> &mut Self::Scalar;
    fn y_mut(&mut self) -> &mut Self::Scalar;

    fn from_xy(x: Self::Scalar, y: Self::Scalar) -> Self;

    fn xy(&self) -> (Self::Scalar, Self::Scalar) {
        (self.x(), self.y())
    }
}

impl<T: CoordNum> Coord for Coordinate<T> {
    type Scalar = T;

    fn x(&self) -> T {
        self.x
    }

    fn y(&self) -> T {
        self.y
    }

    fn x_mut(&mut self) -> &mut T {
        &mut self.x
    }

    fn y_mut(&mut self) -> &mut T {
        &mut self.y
    }

    fn xy(&self) -> (T, T) {
        self.x_y()
    }

    fn from_xy(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T: CoordNum> Coord for Point<T> {
    type Scalar = T;

    fn x(&self) -> T {
        self.0.x()
    }

    fn y(&self) -> T {
        self.0.y()
    }

    fn x_mut(&mut self) -> &mut T {
        &mut self.0.x
    }

    fn y_mut(&mut self) -> &mut T {
        &mut self.0.y
    }

    fn xy(&self) -> (T, T) {
        self.0.x_y()
    }

    fn from_xy(x: T, y: T) -> Self {
        Self::new(x, y)
    }
}
