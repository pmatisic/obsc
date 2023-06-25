# Computational Modeling Challenge 2023
# Hrvoje Abraham, AVL-AST d.o.o., 11.05.2023.
#
# Windows 11 Pro 10.0.22621, Julia 1.9.0 64-bit, CRlibm v1.0.1, Optim v1.7.5
# usage: julia provjera.jl rjesenje.csv

using Combinatorics
using CRlibm
using CSV
using DataFrames
using Optim
using Printf

function validate_solution(data)
    NUMBER_OF_ELLIPSES = 200
    NUMBER_OF_BIG_ELLIPSES = 39
    NUMBER_OF_SMALL_ELLIPSES = 161

    # ellipse axes dimensions in 1/1000th of mm
    BIG_ELLIPSE_AXIS_A = 40_000
    BIG_ELLIPSE_AXIS_B = 20_000
    LITTLE_ELLIPSE_AXIS_A = 20_000
    LITTLE_ELLIPSE_AXIS_B = 10_000

    # rotation angle upper limit in 1/1000th of degree
    ROTATION_ANGLE_UPPER_LIMIT = 179_999

    if size(data) != (NUMBER_OF_ELLIPSES, 5)
        @printf "ERROR: Input file does not contain 5 columns and 200 rows!"
        return false
    end

    if eltype.(eachcol(data)) != DataType[Int64, Int64, Int64, Int64, Int64]
        @printf "ERROR: Input file does not contain only integers!"
        return false
    end

    big_ellipse_count = 0
    little_ellipse_count = 0
    for (i, (_, _, axis_a, axis_b, angle)) in enumerate(eachrow(data))
        if axis_a == LITTLE_ELLIPSE_AXIS_A && axis_b == LITTLE_ELLIPSE_AXIS_B
            little_ellipse_count += 1
        elseif axis_a == BIG_ELLIPSE_AXIS_A && axis_b == BIG_ELLIPSE_AXIS_B
            big_ellipse_count += 1
        else
            @printf "ERROR: Ellipse %d has incorrect axes sizes %d, %d!" i axis_a axis_b
            return false
        end

        if !( 0 <= angle <= ROTATION_ANGLE_UPPER_LIMIT )
            @printf "ERROR: Rotation angle of ellipse %d is out of range!" i
            return false
        end
    end

    if big_ellipse_count != NUMBER_OF_BIG_ELLIPSES
        @printf "ERROR: Expected number of big ellipses is 39, %d were found!" big_ellipse_count
        return false
    end

    if little_ellipse_count != NUMBER_OF_SMALL_ELLIPSES
        @printf "ERROR: Expected number of little ellipses is 161, %d were found!" little_ellipse_count
        return false
    end

    return true
end

function preprocess_ellipse(i, center_x, center_y, axis_a, axis_b, angle)
    # scale dimensions to values close to 1 without loss of precision
    center_x /= 2^16
    center_y /= 2^16

    axis_a /= 2^16
    axis_b /= 2^16

    # 1000ths of degree into radians
    angle = deg2rad(angle / 1000)

    # use correctly-rounded methods to be sure the score is accurate
    crsin = CRlibm.sin(angle)
    crcos = CRlibm.cos(angle)

    # ellipse bounding box half-width and half-height
    box_w = hypot(axis_a * crcos, axis_b * crsin)
    box_h = hypot(axis_a * crsin, axis_b * crcos)

    # precalculate matrix S as defined in the reference and used in check_ellipse_overlap(...)
    R = [1 / (axis_a * axis_a)  0; 0  1 / (axis_b * axis_b)]
    U = [crcos  crsin; crsin  -crcos]
    S = inv(U * R * transpose(U))

    return [i, [center_x, center_y], center_x - box_w, center_x + box_w, center_y - box_h, center_y + box_h, S]
end

function check_ellipse_overlap(ellipse_a, ellipse_b)
    # Ellipse overlap calculation based on:
    #
    # Gilitschenski, Igor, and Uwe D. Hanebeck. "A robust computational test for overlap of two
    # arbitrary-dimensional ellipsoids in fault-detection of kalman filters." 2012 15th
    # International Conference on Information Fusion. IEEE, 2012.

    _, center_a, _, _, _, _, Sa = ellipse_a
    _, center_b, _, _, _, _, Sb = ellipse_b

    v = center_a - center_b
    tv = transpose(v)

    K(s) = 1 - tv * inv(Sa / (1 - s) + Sb / s) * v

    return minimum(optimize(K, 0, 1)) > 0
end

function find_overlap(data)
    for (ellipse_a, ellipse_b) in combinations(data, 2)
        i, _, xa_min, xa_max, ya_min, ya_max, _ = ellipse_a
        j, _, xb_min, xb_max, yb_min, yb_max, _ = ellipse_b

        #  if bounding boxes don't overlap, then ellipses surely don't either
        if xa_min >= xb_max || xa_max <= xb_min || ya_min >= yb_max || ya_max <= yb_min
            continue
        end

        if check_ellipse_overlap(ellipse_a, ellipse_b)
            return true, i, j
        end
    end

    return false, nothing, nothing
end

function evaluate_solution(data)
    x_min, x_max = Inf, -Inf
    y_min, y_max = Inf, -Inf

    for (_, _, bx_min, bx_max, by_min, by_max, _) in data
        x_min = min(x_min, bx_min)
        x_max = max(x_max, bx_max)
        y_min = min(y_min, by_min)
        y_max = max(y_max, by_max)
    end

    # scale the result back into the original units
    return max(x_max - x_min, y_max - y_min) * 2^16
end

function main()
    data = CSV.File(ARGS[1]; header = 0, delim = ",") |> DataFrame

    if !validate_solution(data)
        return
    end

    preprocessed_data = []
    for (i, (center_x, center_y, axis_a, axis_b, angle)) in enumerate(eachrow(data))
        push!(preprocessed_data, preprocess_ellipse(i, center_x, center_y, axis_a, axis_b, angle))
    end

    contains_overlap, i, j = find_overlap(preprocessed_data)
    if contains_overlap
        @printf "ERROR: Ellipses %d and %d are OVERLAPPING!" i j
        return
    end
    @printf "Solution format is correct and no overlaps were found.\n\n"

    @printf "Solution score = %.17g" evaluate_solution(preprocessed_data)
end

main()
