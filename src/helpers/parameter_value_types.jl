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

struct TimePatternValue
    dict::Dict{TimePattern,T} where T
    default
end

struct TimeSeriesValue{I,V,DV}
    time_stamps::I
    values::Array{V,1}
    default::DV
    ignore_year::Bool
    repeat::Bool
    span::Period
    mean_value::V
    function TimeSeriesValue(i::I, v::Array{V,1}, d::DV, iy=false, r=false) where {I,V,DV}
        if length(i) != length(v)
            error("lengths don't match")
        else
            if r
                # Compute span and mean value to save work when accessing repeating time series
                s = i[end] - i[1]
                mv = mean(v)
            else
                s = zero(Hour)
                mv = 0
            end
            new{I,V,DV}(i, v, d, iy, r, s, mv)
        end
    end
end

function TimeSeriesValue(db_value::Dict, default)
    if !haskey(db_value, "data")
        # Naked dict, no meta
        data = db_value
        metadata = Dict()
    else
        data = db_value["data"]
        metadata = get(db_value, "metadata", Dict())
    end
    TimeSeriesValue(data, metadata, default)
end

function TimeSeriesValue(db_value::Array, default)
    # Naked array, no meta
    metadata = Dict()
    TimeSeriesValue(db_value, metadata, default)
end

function TimeSeriesValue(data::Dict, metadata::Dict, default)
    # time_stamps come with data, so just look for "ignore_year" in metadata
    repeat = false
    ignore_year = get(metadata, "ignore_year", false)
    data = Dict(parse_date_time(k) => v for (k, v) in data)
    ignore_year && (data = Dict(k - Year(k) => v for (k, v) in data))
    data = sort(data)
    TimeSeriesValue(collect(keys(data)), collect(values(data)), default, ignore_year, repeat)
end

function TimeSeriesValue(data::Array, metadata::Dict, default)
    # Look at the first element in data to see whether it's one column or two column format (and pray)
    if data[1] isa Array
        # Two column array format
        TimeSeriesValue(Dict(k => v for (k, v) in data), metadata, default)
    else
        # One column array format
        if haskey(metadata, "start")
            start = parse_date_time(metadata["start"])
            ignore_year = get(metadata, "ignore_year", false)
            repeat = get(metadata, "repeat", false)
        else
            start = DateTime(1)
            ignore_year = get(metadata, "ignore_year", true)
            repeat = get(metadata, "repeat", true)
        end
        ignore_year && (start -= Year(start))
        len = length(data) - 1
        if haskey(metadata, "resolution")
            resolution = metadata["resolution"]
            if resolution isa Array
                rlen = length(resolution)
                if rlen > len
                    # Trim
                    resolution = resolution[1:len]
                elseif rlen < len
                    # Repeat
                    ratio = div(len, rlen)
                    tail_len = len - ratio * rlen
                    tail = resolution[1:tail_len]
                    resolution = vcat(repeat(resolution, ratio), tail)
                end
                res = parse_duration.(resolution)
                inds = cumsum(vcat(start, res))
            else
                res = parse_duration(resolution)
                end_ = start + len * res
                inds = start:res:end_
            end
        else
            res = Hour(1)
            end_ = start + len * res
            inds = start:res:end_
        end
        TimeSeriesValue(inds, data, default, ignore_year, repeat)
    end
end

function (p::TimePatternValue)(;t::Union{TimeSlice,Nothing}=nothing)
    t === nothing && error("argument `t` missing")
    values = [val for (tp, val) in p.dict if match(t, tp)]
    if isempty(values)
        @warn("$t does not match $p, using default value...")
        p.default
    else
        mean(values)
    end
end

function (p::TimeSeriesValue)(;t::Union{TimeSlice,Nothing}=nothing)
    t === nothing && return p
    start = t.start
    end_ = t.end_
    duration = t.duration
    if p.ignore_year
        start -= Year(start)
        end_ = start + duration
    end
    if p.repeat
        repetitions = 0
        if start > p.time_stamps[end]
            # Move start back within time_stamps range
            mismatch = start - p.time_stamps[1]
            repetitions = div(mismatch, p.span)
            start -= repetitions * p.span
            end_ = start + duration
        end
        if end_ > p.time_stamps[end]
            # Move end_ back within time_stamps range
            mismatch = end_ - p.time_stamps[1]
            repetitions = div(mismatch, p.span)
            end_ -= repetitions * p.span
        end
        a = findfirst(i -> i >= start, p.time_stamps)
        b = findlast(i -> i < end_, p.time_stamps)
        if a === nothing || b === nothing
            @warn("$p is not defined on $t, using default value...")
            p.default
        else
            if a <= b
                value = mean(p.values[a:b])
            else
                value = -mean(p.values[b:a])
            end
            value + repetitions * p.mean_value  # repetitions holds the number of rolls we move back the end
        end
    else
        a = findfirst(i -> i >= start, p.time_stamps)
        b = findlast(i -> i <= end_, p.time_stamps)
        if a === nothing || b === nothing
            @warn("$p is not defined on $t, using default value...")
            p.default
        else
            mean(p.values[a:b])
        end
    end
end

time_stamps(val::TimeSeriesValue) = val.time_stamps
