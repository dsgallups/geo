use crate::geometry::*;
use crate::CoordNum;

pub trait GeometryTrait: PartialEq {
    type Coord: CoordTrait;
}

impl<T: CoordNum> GeometryTrait for Point<T> {
    type Coord = Coordinate<T>;
}
impl<T: CoordNum> GeometryTrait for Line<T> {
    type Coord = Coordinate<T>;
}
impl<T: CoordNum> GeometryTrait for LineString<T> {
    type Coord = Coordinate<T>;
}
impl<T: CoordNum> GeometryTrait for Polygon<T> {
    type Coord = Coordinate<T>;
}
impl<T: CoordNum> GeometryTrait for MultiPoint<T> {
    type Coord = Coordinate<T>;
}
impl<T: CoordNum> GeometryTrait for MultiLineString<T> {
    type Coord = Coordinate<T>;
}
impl<T: CoordNum> GeometryTrait for MultiPolygon<T> {
    type Coord = Coordinate<T>;
}
impl<T: CoordNum> GeometryTrait for Rect<T> {
    type Coord = Coordinate<T>;
}
impl<T: CoordNum> GeometryTrait for Triangle<T> {
    type Coord = Coordinate<T>;
}
impl<T: CoordNum> GeometryTrait for GeometryCollection<T> {
    type Coord = Coordinate<T>;
}
impl<T: CoordNum> GeometryTrait for Geometry<T> {
    type Coord = Coordinate<T>;
}

pub trait CoordTrait: PartialEq {
    type Scalar: CoordNum;

    fn x(&self) -> Self::Scalar;
    fn y(&self) -> Self::Scalar;
    fn x_mut(&mut self) -> &mut Self::Scalar;
    fn y_mut(&mut self) -> &mut Self::Scalar;

    fn xy(&self) -> (Self::Scalar, Self::Scalar);
    fn from_xy(x: Self::Scalar, y: Self::Scalar) -> Self;
}

impl<T: CoordNum> CoordTrait for Coordinate<T> {
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
