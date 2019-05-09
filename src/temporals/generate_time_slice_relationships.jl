#############################################################################
# Copyright (C) 2017 - 2018  Spine Project
#
# This file is part of Spine Model.
#
# Spine Model is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spine Model is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#############################################################################
# TODO: have an eye on where unique! is necessary for speedup
# TODO: add examples to all docstrings when all this begins to converge

struct TBeforeTFunction
    list::Array{Tuple{TimeSlice,TimeSlice},1}
end

struct TInTFunction
    list::Array{Tuple{TimeSlice,TimeSlice},1}
end

struct TInTExclFunction
    list::Array{Tuple{TimeSlice,TimeSlice},1}
end

struct TOverlapsTFunction
    list::Array{Tuple{TimeSlice,TimeSlice},1}
end

struct TOverlapsTExclFunction
    list::Array{Tuple{TimeSlice,TimeSlice},1}
end

"""
    t_before_t(;t_before=nothing, t_after=nothing)

Return the list of tuples `(t1, t2)` where `t1` is right before `t2` in the sense that it
ends when `t2` starts, i.e. `t1.end_ == t2.start`.
If `t_before` is not `nothing`, return the list of time slices that succeed `t_before`
(or any element in `t_before` if it's a list).
If `t_after` is not `nothing`, return the list of time slices that are succeeded by `t_after`
(or any element in `t_after` if it's a list).
"""
function (t_before_t::TBeforeTFunction)(;t_before=nothing, t_after=nothing)
    result = t_before_t.list
    if t_before != nothing
        result = [(t1, t2) for (t1, t2) in result if t1 in tuple(t_before...)]
    end
    if t_after != nothing
        result = [(t1, t2) for (t1, t2) in result if t2 in tuple(t_after...)]
    end
    if t_before != nothing && t_after == nothing
        [t2 for (t1, t2) in result]
    elseif t_before == nothing && t_after != nothing
        [t1 for (t1, t2) in result]
    else
        result
    end
end

"""
    t_in_t(;t_short=nothing, t_long=nothing)

Return the list of tuples `(t1, t2)`, where `t2` is contained in `t1`.
If `t_long` is not `nothing`, return the list of time slices contained in `t_long`
(or any element in `t_long` if it's a list).
If `t_short` is not `nothing`, return the list of time slices that contain `t_short`
(or any element in `t_short` if it's a list).
"""
function (t_in_t::TInTFunction)(;t_short=nothing, t_long=nothing)
    result = t_in_t.list
    if t_short != nothing
        result = [(t1, t2) for (t1, t2) in result if t1 in tuple(t_short...)]
    end
    if t_long != nothing
        result = [(t1, t2) for (t1, t2) in result if t2 in tuple(t_long...)]
    end
    if t_short != nothing && t_long == nothing
        [t2 for (t1, t2) in result]
    elseif t_short == nothing && t_long != nothing
        [t1 for (t1, t2) in result]
    else
        result
    end
end

"""
    t_in_t_excl(;t_short=nothing, t_long=nothing)

Return the list of tuples `(t1, t2)`, where `t1` contains `t2` and `t1` is different from `t2`.
See [`t_in_t(;t_long=nothing, t_short=nothing)`](@ref)
for details about keyword arguments `t_long`, `t_short`.
"""
function (t_in_t_excl::TInTExclFunction)(;t_short=nothing, t_long=nothing)
    result = t_in_t_excl.list
    if t_short != nothing
        result = [(t1, t2) for (t1, t2) in result if t1 in tuple(t_short...)]
    end
    if t_long != nothing
        result = [(t1, t2) for (t1, t2) in result if t2 in tuple(t_long...)]
    end
    if t_short != nothing && t_long == nothing
        [t2 for (t1, t2) in result]
    elseif t_short == nothing && t_long != nothing
        [t1 for (t1, t2) in result]
    else
        result
    end
end

"""
    t_overlaps_t()

Return the list of tuples `(t1, t2)` where `t1` and `t2` have some time in common.
"""
function (t_overlaps_t::TOverlapsTFunction)()
    t_overlaps_t.list
end

"""
    t_overlaps_t(t_overlap)

Return the list of time slices that have some time in common with `t_overlap`
(or some time in common with any element in `t_overlap` if it's a list).
"""
function (t_overlaps_t::TOverlapsTFunction)(t_overlap)
    unique(t2 for (t1, t2) in t_overlaps_t.list if t1 in tuple(t_overlap...))
end

"""
    t_overlaps_t(t_list1, t_list2)

Return a list of time slices which are in `t_list1` and have some time in common
with any of the time slices in `t_list2` and vice versa.
"""
function (t_overlaps_t::TOverlapsTFunction)(t_list1, t_list2)
    orig_list = t_overlaps_t.list
    overlap_list = [
        (t1, t2) for (t1, t2) in orig_list if t1 in tuple(t_list1...) && t2 in tuple(t_list2...)
    ]
    unique(vcat(first.(overlap_list), last.(overlap_list)))
end

"""
    t_overlaps_t_excl()

Return the list of tuples `(t1, t2)` where `t1` and `t2` have some time in common
and `t1` is not equal to `t2`.
"""
function (t_overlaps_t_excl::TOverlapsTExclFunction)()
    t_overlaps_t_excl.list
end


"""
    t_overlaps_t_excl(t_overlap)

Return the list of time slices that have some time in common with `t_overlap`
(or some time in common with any element in `t_overlap` if it's a list) and `t1` is not equal to `t2`.
"""
function (t_overlaps_t_excl::TOverlapsTExclFunction)(t_overlap)
    unique(t2 for (t1, t2) in t_overlaps_t_excl.list if t1 in tuple(t_overlap...))
end
"""
    t_overlaps_t_excl(t_list1, t_list2)

Return a list of time slices which are in `t_list1` and have some time in common
with any of the time slices in `t_list2` (unless they are the same time slice) and vice versa.
"""
function (t_overlaps_t_excl::TOverlapsTExclFunction)(t_list1, t_list2)
    orig_list = t_overlaps_t_excl.list
    overlap_list = [
        (t1, t2) for (t1, t2) in orig_list if t1 in tuple(t_list1...) && t2 in tuple(t_list2...)
    ]
    unique(vcat(first.(overlap_list), last.(overlap_list)))
end

"""
    generate_time_slice_relationships()

Create and export convenience functions to access time slice relationships:
`t_in_t`, `t_preceeds_t`, `t_overlaps_t`...
"""
function generate_time_slice_relationships()
    t_before_t_list = []
    t_in_t_list = []
    t_overlaps_t_list = []
    for i in time_slice()
        for j in time_slice()
            if succeeds(j, i)
                push!(t_before_t_list, tuple(i, j))
            end
            if in(i, j)
                push!(t_in_t_list, tuple(i, j))
            end
            if overlaps(i, j)
                push!(t_overlaps_t_list, tuple(i, j))
            end
        end
    end
    # TODO: instead of unique -> check beforehand whether timeslice tuple is already added
    # Is `unique!()` slow? I fear the above check can be a bit slow.
    # An alternative is to use `Set()` instead of `[]` to warranty uniqueness,
    # but then we lose the order - do we care about order?
    unique!(t_in_t_list)
    unique!(t_overlaps_t_list)
    t_in_t_excl_list = [(t1, t2) for (t1, t2) in t_in_t_list if t1 != t2]
    t_overlaps_t_excl_list = [(t1, t2) for (t1, t2) in t_overlaps_t_list if t1 != t2]
    # Create function-like objects
    t_before_t = TBeforeTFunction(t_before_t_list)
    t_in_t = TInTFunction(t_in_t_list)
    t_in_t_excl = TInTExclFunction(t_in_t_excl_list)
    t_overlaps_t = TOverlapsTFunction(t_overlaps_t_list)
    t_overlaps_t_excl = TOverlapsTExclFunction(t_overlaps_t_excl_list)
    # Export the function-like objects
    @eval begin
        t_before_t = $t_before_t
        t_in_t = $t_in_t
        t_in_t_excl = $t_in_t_excl
        t_overlaps_t = $t_overlaps_t
        t_overlaps_t_excl = $t_overlaps_t_excl
        export t_before_t
        export t_in_t
        export t_in_t_excl
        export t_overlaps_t
        export t_overlaps_t_excl
    end
end
