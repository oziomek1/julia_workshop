#= basics:
- Julia version: 0.6
- Author: oziomek -
Date: 2018-03-22 =#
mutable struct Point{T<:Real}
    x::T
    y::T
end

mutable struct Rectangle
    a::Point
    b::Point
    c::Point
    d::Point
end

function initializePoints()
    point1 = Point(1.1, 2.2)
    point2 = Point(1.7, 2.1)
    point3 = Point(8.4, 2.6)
    point4 = Point(5.4, 3.7)
    points_array = [point1, point2, point3, point4]
    for point in points_array
        println(point)
    end
    return point1, point2, point3, point4
end
function createRectangle()
    p1, p2, p3, p4 = initializePoints()
    initializePoints()
    rectangle = Rectangle(p1, p2, p3, p4)

    println(rectangle)
end

createRectangle()
