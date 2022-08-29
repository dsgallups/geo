use crate::num_traits::{One, Zero};
use crate::traits::CoordTrait;
use crate::{CoordFloat, CoordNum, Coordinate, MapCoords, MapCoordsInPlace};

use std::fmt;

/// Apply an [`AffineTransform`] like [`scale`](AffineTransform::scale),
/// [`skew`](AffineTransform::skew), or [`rotate`](AffineTransform::rotate) to a
/// [`Geometry`](crate::geometry::Geometry).
///
/// Multiple transformations can be composed in order to be efficiently applied in a single
/// operation. See [`AffineTransform`] for more on how to build up a transformation.
///
/// If you are not composing operations, traits that leverage this same machinery exist which might
/// be more readable. See: [`Scale`](crate::algorithm::Scale),
/// [`Translate`](crate::algorithm::Translate), [`Rotate`](crate::algorithm::Rotate),
/// and [`Skew`](crate::algorithm::Skew).
///
/// # Examples
/// ## Build up transforms by beginning with a constructor, then chaining mutation operations
/// ```
/// use geo::{AffineOps, AffineTransform};
/// use geo::{line_string, BoundingRect, Point, LineString};
/// use approx::assert_relative_eq;
///
/// let ls: LineString = line_string![
///     (x: 0.0f64, y: 0.0f64),
///     (x: 0.0f64, y: 10.0f64),
/// ];
/// let center = ls.bounding_rect().unwrap().center();
///
/// let transform = AffineTransform::skew(40.0, 40.0, center).rotated(45.0, center);
///
/// let skewed_rotated = ls.affine_transform(&transform);
///
/// assert_relative_eq!(skewed_rotated, line_string![
///     (x: 0.5688687f64, y: 4.4311312),
///     (x: -0.5688687, y: 5.5688687)
/// ], max_relative = 1.0);
/// ```
pub trait AffineOps {
    type Coord: CoordTrait;

    /// Apply `transform` immutably, outputting a new geometry.
    #[must_use]
    fn affine_transform(&self, transform: &AffineTransform<Self::Coord>) -> Self;

    /// Apply `transform` to mutate `self`.
    fn affine_transform_mut(&mut self, transform: &AffineTransform<Self::Coord>);
}

impl<C: CoordTrait, M: MapCoordsInPlace<Coord = C> + MapCoords<C, InCoord = C, Output = Self>>
    AffineOps for M
{
    type Coord = C;

    fn affine_transform(&self, transform: &AffineTransform<Self::Coord>) -> Self {
        self.map_coords(|c| transform.apply(c))
    }

    fn affine_transform_mut(&mut self, transform: &AffineTransform<Self::Coord>) {
        self.map_coords_in_place(|c| transform.apply(c))
    }
}

/// A general affine transformation matrix, and associated operations.
///
/// Note that affine ops are **already implemented** on most `geo-types` primitives, using this module.
///
/// Affine transforms using the same numeric type (e.g. [`CoordFloat`](crate::CoordFloat)) can be **composed**,
/// and the result can be applied to geometries using e.g. [`MapCoords`](crate::MapCoords). This allows the
/// efficient application of transforms: an arbitrary number of operations can be chained.
/// These are then composed, producing a final transformation matrix which is applied to the geometry coordinates.
///
/// `AffineTransform` is a row-major matrix.
/// 2D affine transforms require six matrix parameters:
///
/// `[a, b, xoff, d, e, yoff]`
///
/// these map onto the `AffineTransform` rows as follows:
/// ```ignore
/// [[a, b, xoff],
/// [d, e, yoff],
/// [0, 0, 1]]
/// ```
/// The equations for transforming coordinates `(x, y) -> (x', y')` are given as follows:
///
/// `x' = ax + by + xoff`
///
/// `y' = dx + ey + yoff`
///
/// # Usage
///
/// Two types of operation are provided: construction and mutation. **Construction** functions create a *new* transform
/// and are denoted by the use of the **present tense**: `scale()`, `translate()`, `rotate()`, and `skew()`.
///
/// **Mutation** methods *add* a transform to the existing `AffineTransform`, and are denoted by the use of the past participle:
/// `scaled()`, `translated()`, `rotated()`, and `skewed()`.
///
/// # Examples
/// ## Build up transforms by beginning with a constructor, then chaining mutation operations
/// ```
/// use geo::{AffineOps, AffineTransform};
/// use geo::{line_string, BoundingRect, Point, LineString};
/// use approx::assert_relative_eq;
///
/// let ls: LineString = line_string![
///     (x: 0.0f64, y: 0.0f64),
///     (x: 0.0f64, y: 10.0f64),
/// ];
/// let center = ls.bounding_rect().unwrap().center();
///
/// let transform = AffineTransform::skew(40.0, 40.0, center).rotated(45.0, center);
///
/// let skewed_rotated = ls.affine_transform(&transform);
///
/// assert_relative_eq!(skewed_rotated, line_string![
///     (x: 0.5688687f64, y: 4.4311312),
///     (x: -0.5688687, y: 5.5688687)
/// ], max_relative = 1.0);
/// ```
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct AffineTransform<C: CoordTrait = Coordinate<f64>> {
    matrix: [[C::Scalar; 3]; 3],
}

impl<C: CoordTrait> Default for AffineTransform<C> {
    fn default() -> Self {
        // identity matrix
        Self::identity()
    }
}

impl<C: CoordTrait> AffineTransform<C> {
    /// Create a new affine transformation by composing two `AffineTransform`s.
    ///
    /// This is a **cumulative** operation; the new transform is *added* to the existing transform.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        // lol
        let matrix = [
            [
                (self.matrix[0][0] * other.matrix[0][0])
                    + (self.matrix[0][1] * other.matrix[1][0])
                    + (self.matrix[0][2] * other.matrix[2][0]),
                (self.matrix[0][0] * other.matrix[0][1])
                    + (self.matrix[0][1] * other.matrix[1][1])
                    + (self.matrix[0][2] * other.matrix[2][1]),
                (self.matrix[0][0] * other.matrix[0][2])
                    + (self.matrix[0][1] * other.matrix[1][2])
                    + (self.matrix[0][2] * other.matrix[2][2]),
            ],
            [
                (self.matrix[1][0] * other.matrix[0][0])
                    + (self.matrix[1][1] * other.matrix[1][0])
                    + (self.matrix[1][2] * other.matrix[2][0]),
                (self.matrix[1][0] * other.matrix[0][1])
                    + (self.matrix[1][1] * other.matrix[1][1])
                    + (self.matrix[1][2] * other.matrix[2][1]),
                (self.matrix[1][0] * other.matrix[0][2])
                    + (self.matrix[1][1] * other.matrix[1][2])
                    + (self.matrix[1][2] * other.matrix[2][2]),
            ],
            [
                // this section isn't technically necessary since the last row is invariant: [0, 0, 1]
                (self.matrix[2][0] * other.matrix[0][0])
                    + (self.matrix[2][1] * other.matrix[1][0])
                    + (self.matrix[2][2] * other.matrix[2][0]),
                (self.matrix[2][0] * other.matrix[0][1])
                    + (self.matrix[2][1] * other.matrix[1][1])
                    + (self.matrix[2][2] * other.matrix[2][1]),
                (self.matrix[2][0] * other.matrix[0][2])
                    + (self.matrix[2][1] * other.matrix[1][2])
                    + (self.matrix[2][2] * other.matrix[2][2]),
            ],
        ];
        Self { matrix }
    }
    /// Create the identity matrix
    ///
    /// The matrix is:
    /// ```ignore
    /// [[1, 0, 0],
    /// [0, 1, 0],
    /// [0, 0, 1]]
    /// ```
    pub fn identity() -> Self {
        Self::new(
            C::Scalar::one(),
            C::Scalar::zero(),
            C::Scalar::zero(),
            C::Scalar::zero(),
            C::Scalar::one(),
            C::Scalar::zero(),
        )
    }

    /// Whether the transformation is equivalent to the [identity matrix](Self::identity),
    /// that is, whether it's application will be a a no-op.
    ///
    /// ```
    /// use geo::AffineTransform;
    /// let mut transform = AffineTransform::identity();
    /// assert!(transform.is_identity());
    ///
    /// // mutate the transform a bit
    /// transform = transform.translated(1.0, 2.0);
    /// assert!(!transform.is_identity());
    ///
    /// // put it back
    /// transform = transform.translated(-1.0, -2.0);
    /// assert!(transform.is_identity());
    /// ```
    pub fn is_identity(&self) -> bool {
        self == &Self::identity()
    }

    /// **Create** a new affine transform for scaling, scaled by factors along the `x` and `y` dimensions.
    /// The point of origin is *usually* given as the 2D bounding box centre of the geometry, but
    /// any coordinate may be specified.
    /// Negative scale factors will mirror or reflect coordinates.
    ///
    /// The matrix is:
    /// ```ignore
    /// [[xfact, 0, xoff],
    /// [0, yfact, yoff],
    /// [0, 0, 1]]
    ///
    /// xoff = origin.x - (origin.x * xfact)
    /// yoff = origin.y - (origin.y * yfact)
    /// ```
    pub fn scale(
        xfact: C::Scalar,
        yfact: C::Scalar,
        origin: impl Into<Coordinate<C::Scalar>>,
    ) -> Self {
        let (x0, y0) = origin.into().x_y();
        let xoff = x0 - (x0 * xfact);
        let yoff = y0 - (y0 * yfact);
        Self::new(
            xfact,
            C::Scalar::zero(),
            xoff,
            C::Scalar::zero(),
            yfact,
            yoff,
        )
    }

    /// **Add** an affine transform for scaling, scaled by factors along the `x` and `y` dimensions.
    /// The point of origin is *usually* given as the 2D bounding box centre of the geometry, but
    /// any coordinate may be specified.
    /// Negative scale factors will mirror or reflect coordinates.
    /// This is a **cumulative** operation; the new transform is *added* to the existing transform.
    #[must_use]
    pub fn scaled(
        mut self,
        xfact: C::Scalar,
        yfact: C::Scalar,
        origin: impl Into<Coordinate<C::Scalar>>,
    ) -> Self {
        self.matrix = self.compose(&Self::scale(xfact, yfact, origin)).matrix;
        self
    }

    /// **Create** an affine transform for translation, shifted by offsets along the `x` and `y` dimensions.
    ///
    /// The matrix is:
    /// ```ignore
    /// [[1, 0, xoff],
    /// [0, 1, yoff],
    /// [0, 0, 1]]
    /// ```
    pub fn translate(xoff: C::Scalar, yoff: C::Scalar) -> Self {
        Self::new(
            C::Scalar::one(),
            C::Scalar::zero(),
            xoff,
            C::Scalar::zero(),
            C::Scalar::one(),
            yoff,
        )
    }

    /// **Add** an affine transform for translation, shifted by offsets along the `x` and `y` dimensions
    ///
    /// This is a **cumulative** operation; the new transform is *added* to the existing transform.
    #[must_use]
    pub fn translated(mut self, xoff: C::Scalar, yoff: C::Scalar) -> Self {
        self.matrix = self.compose(&Self::translate(xoff, yoff)).matrix;
        self
    }

    /// Apply the current transform to a coordinate
    pub fn apply(&self, coord: C) -> C {
        C::from_xy(
            self.matrix[0][0] * coord.x() + self.matrix[0][1] * coord.y() + self.matrix[0][2],
            self.matrix[1][0] * coord.x() + self.matrix[1][1] * coord.y() + self.matrix[1][2],
        )
    }

    /// Create a new custom transform matrix
    ///
    /// The argument order matches that of the affine transform matrix:
    ///```ignore
    /// [[a, b, xoff],
    /// [d, e, yoff],
    /// [0, 0, 1]] <-- not part of the input arguments
    /// ```
    pub fn new(
        a: C::Scalar,
        b: C::Scalar,
        xoff: C::Scalar,
        d: C::Scalar,
        e: C::Scalar,
        yoff: C::Scalar,
    ) -> Self {
        let matrix = [
            [a, b, xoff],
            [d, e, yoff],
            [C::Scalar::zero(), C::Scalar::zero(), C::Scalar::one()],
        ];
        Self { matrix }
    }
}

impl<T: CoordNum> fmt::Debug for AffineTransform<Coordinate<T>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AffineTransform")
            .field("a", &self.matrix[0][0])
            .field("b", &self.matrix[0][1])
            .field("xoff", &self.matrix[1][2])
            .field("d", &self.matrix[1][0])
            .field("e", &self.matrix[1][1])
            .field("yoff", &self.matrix[1][2])
            .finish()
    }
}

impl<T: CoordNum> From<[T; 6]> for AffineTransform<Coordinate<T>> {
    fn from(arr: [T; 6]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5])
    }
}

impl<T: CoordNum> From<(T, T, T, T, T, T)> for AffineTransform<Coordinate<T>> {
    fn from(tup: (T, T, T, T, T, T)) -> Self {
        Self::new(tup.0, tup.1, tup.2, tup.3, tup.4, tup.5)
    }
}

impl<U: CoordFloat, C: CoordTrait<Scalar = U>> AffineTransform<C> {
    /// **Create** an affine transform for rotation, using an arbitrary point as its centre.
    ///
    /// Note that this operation is only available for geometries with floating point coordinates.
    ///
    /// `angle` is given in **degrees**.
    ///
    /// The matrix (angle denoted as theta) is:
    /// ```ignore
    /// [[cos_theta, -sin_theta, xoff],
    /// [sin_theta, cos_theta, yoff],
    /// [0, 0, 1]]
    ///
    /// xoff = origin.x - (origin.x * cos(theta)) + (origin.y * sin(theta))
    /// yoff = origin.y - (origin.x * sin(theta)) + (origin.y * cos(theta))
    /// ```
    pub fn rotate(degrees: U, origin: impl Into<Coordinate<U>>) -> Self {
        let (sin_theta, cos_theta) = degrees.to_radians().sin_cos();
        let (x0, y0) = origin.into().x_y();
        let xoff = x0 - (x0 * cos_theta) + (y0 * sin_theta);
        let yoff = y0 - (x0 * sin_theta) - (y0 * cos_theta);
        Self::new(cos_theta, -sin_theta, xoff, sin_theta, cos_theta, yoff)
    }

    /// **Add** an affine transform for rotation, using an arbitrary point as its centre.
    ///
    /// Note that this operation is only available for geometries with floating point coordinates.
    ///
    /// `angle` is given in **degrees**.
    ///
    /// This is a **cumulative** operation; the new transform is *added* to the existing transform.
    #[must_use]
    pub fn rotated(mut self, angle: U, origin: impl Into<Coordinate<U>>) -> Self {
        self.matrix = self.compose(&Self::rotate(angle, origin)).matrix;
        self
    }

    /// **Create** an affine transform for skewing.
    ///
    /// Note that this operation is only available for geometries with floating point coordinates.
    ///
    /// Geometries are sheared by angles along x (`xs`) and y (`ys`) dimensions.
    /// The point of origin is *usually* given as the 2D bounding box centre of the geometry, but
    /// any coordinate may be specified. Angles are given in **degrees**.
    /// The matrix is:
    /// ```ignore
    /// [[1, tan(x), xoff],
    /// [tan(y), 1, yoff],
    /// [0, 0, 1]]
    ///
    /// xoff = -origin.y * tan(xs)
    /// yoff = -origin.x * tan(ys)
    /// ```
    pub fn skew(xs: U, ys: U, origin: impl Into<Coordinate<U>>) -> Self {
        let Coordinate { x: x0, y: y0 } = origin.into();
        let mut tanx = xs.to_radians().tan();
        let mut tany = ys.to_radians().tan();
        // These checks are stolen from Shapely's implementation -- may not be necessary
        if tanx.abs() < U::from::<f64>(2.5e-16).unwrap() {
            tanx = U::zero();
        }
        if tany.abs() < U::from::<f64>(2.5e-16).unwrap() {
            tany = U::zero();
        }
        let xoff = -y0 * tanx;
        let yoff = -x0 * tany;
        Self::new(U::one(), tanx, xoff, tany, U::one(), yoff)
    }

    /// **Add** an affine transform for skewing.
    ///
    /// Note that this operation is only available for geometries with floating point coordinates.
    ///
    /// Geometries are sheared by angles along x (`xs`) and y (`ys`) dimensions.
    /// The point of origin is *usually* given as the 2D bounding box centre of the geometry, but
    /// any coordinate may be specified. Angles are given in **degrees**.
    ///
    /// This is a **cumulative** operation; the new transform is *added* to the existing transform.
    #[must_use]
    pub fn skewed(mut self, xs: U, ys: U, origin: impl Into<Coordinate<U>>) -> Self {
        self.matrix = self.compose(&Self::skew(xs, ys, origin)).matrix;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{polygon, Point};

    // given a matrix with the shape
    // [[a, b, xoff],
    // [d, e, yoff],
    // [0, 0, 1]]
    #[test]
    fn matrix_multiply() {
        let a = AffineTransform::<Coordinate<i32>>::new(1, 2, 5, 3, 4, 6);
        let b = AffineTransform::new(7, 8, 11, 9, 10, 12);
        let composed = a.compose(&b);
        assert_eq!(composed.matrix[0][0], 25);
        assert_eq!(composed.matrix[0][1], 28);
        assert_eq!(composed.matrix[0][2], 40);
        assert_eq!(composed.matrix[1][0], 57);
        assert_eq!(composed.matrix[1][1], 64);
        assert_eq!(composed.matrix[1][2], 87);
    }
    #[test]
    fn test_transform_composition() {
        let p0 = Point::new(0.0f64, 0.0);
        // scale once
        let mut scale_a = AffineTransform::<Coordinate<f64>>::default().scaled(2.0, 2.0, p0);
        // rotate
        scale_a = scale_a.rotated(45.0, p0);
        // rotate back
        scale_a = scale_a.rotated(-45.0, p0);
        // scale up again, doubling
        scale_a = scale_a.scaled(2.0, 2.0, p0);
        // scaled once
        let scale_b = AffineTransform::<Coordinate<f64>>::default().scaled(2.0, 2.0, p0);
        // scaled once, but equal to 2 + 2
        let scale_c = AffineTransform::<Coordinate<f64>>::default().scaled(4.0, 4.0, p0);
        assert_ne!(&scale_a.matrix, &scale_b.matrix);
        assert_eq!(&scale_a.matrix, &scale_c.matrix);
    }

    #[test]
    fn affine_transformed() {
        let transform = AffineTransform::translate(1.0, 1.0).scaled(2.0, 2.0, (0.0, 0.0));
        let mut poly = polygon![(x: 0.0, y: 0.0), (x: 0.0, y: 2.0), (x: 1.0, y: 2.0)];
        poly.affine_transform_mut(&transform);

        let expected = polygon![(x: 1.0, y: 1.0), (x: 1.0, y: 5.0), (x: 3.0, y: 5.0)];
        assert_eq!(expected, poly);
    }
}
