# Output: VTK image data for visualisation, plain-text line-outs for quick
# plotting and regression checks, and raw checkpoints for restarts.
#
# The VTK writer emits a self-contained `.vti` file with raw appended binary,
# which ParaView, VisIt and yt all read directly. It is written by hand so that
# the package keeps no dependencies.

const _VTK_ENDIAN = Base.ENDIAN_BOM == 0x04030201 ? "LittleEndian" : "BigEndian"

_vtk_type(::Type{Float64}) = "Float64"
_vtk_type(::Type{Float32}) = "Float32"

"""
    output_fields(sim) -> Vector{Tuple{String,Int,Array}}

The named fields written by [`write_vtk`](@ref): `(name, ncomponents, data)`
where `data` is laid out with the component index first and then `x, y, z`.
"""
function output_fields(sim::Simulation{T}) where {T}
    g = sim.grid
    ir, jr, kr = g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]
    prim = primitives(sim)
    fields = Tuple{String,Int,Array{T}}[]
    push!(fields, ("density", 1, prim[IRHO:IRHO, :, :, :]))
    push!(fields, ("velocity", 3, prim[IVX:IVZ, :, :, :]))
    push!(fields, ("pressure", 1, prim[IPRE:IPRE, :, :, :]))
    push!(fields, ("gas_temperature", 1, reshape(gas_temperature(sim), 1, g.n...)))
    if has_radiation(sim)
        push!(fields, ("radiation_energy", 1, sim.R[IERAD:IERAD, ir, jr, kr]))
        push!(fields, ("radiation_flux", 3, sim.R[IFRX:IFRZ, ir, jr, kr]))
        push!(fields, ("radiation_temperature", 1, reshape(radiation_temperature(sim), 1, g.n...)))
    end
    return fields
end

"""
    write_vtk(sim, path) -> String

Write the interior state as a VTK image-data file (cell data). `path` may be
given with or without the `.vti` extension; the file name actually written is
returned.

A numbered series such as `data_00000.vti`, `data_00001.vti`, ... is recognised
by ParaView as a time series.
"""
function write_vtk(sim::Simulation{T}, path::AbstractString) where {T}
    g = sim.grid
    fn = endswith(path, ".vti") ? String(path) : path * ".vti"
    nx, ny, nz = g.n
    fields = output_fields(sim)

    scalars = ""
    vectors = ""
    for (name, nc, _) in fields
        if nc == 1 && isempty(scalars)
            scalars = name
        elseif nc == 3 && isempty(vectors)
            vectors = name
        end
    end

    io = IOBuffer()
    println(io, "<?xml version=\"1.0\"?>")
    println(io, "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"", _VTK_ENDIAN,
            "\" header_type=\"UInt64\">")
    @printf(io, "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"%.17g %.17g %.17g\" Spacing=\"%.17g %.17g %.17g\">\n",
            nx, ny, nz, g.xlo[1], g.xlo[2], g.xlo[3], g.dx[1], g.dx[2], g.dx[3])
    @printf(io, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n", nx, ny, nz)
    println(io, "      <CellData Scalars=\"", scalars, "\" Vectors=\"", vectors, "\">")
    offset = 0
    for (name, nc, data) in fields
        println(io, "        <DataArray type=\"", _vtk_type(T), "\" Name=\"", name,
                "\" NumberOfComponents=\"", nc, "\" format=\"appended\" offset=\"", offset, "\"/>")
        offset += 8 + length(data) * sizeof(T)
    end
    println(io, "      </CellData>")
    println(io, "    </Piece>")
    println(io, "  </ImageData>")
    println(io, "  <AppendedData encoding=\"raw\">")
    print(io, "   _")
    header = take!(io)

    open(fn, "w") do f
        write(f, header)
        for (_, _, data) in fields
            write(f, UInt64(length(data) * sizeof(T)))
            write(f, vec(data))
        end
        write(f, "\n  </AppendedData>\n</VTKFile>\n")
    end
    return fn
end

"""
    write_line(sim, path; dir = 1, at = nothing)

Write a one-dimensional cut through the domain as a whitespace-separated text
table with a header line: coordinate, then every field of
[`output_fields`](@ref) component by component.

`dir` selects the axis of the cut and `at` the transverse cell indices (a
2-tuple of interior indices, default the middle of the domain). Intended for
quick plotting of shock tubes and Marshak waves.
"""
function write_line(sim::Simulation{T}, path::AbstractString; dir::Int=1, at=nothing) where {T}
    g = sim.grid
    1 <= dir <= 3 || throw(ArgumentError("dir must be 1, 2 or 3"))
    others = (dir == 1 ? (2, 3) : dir == 2 ? (1, 3) : (1, 2))
    idx = at === nothing ? ntuple(m -> cld(g.n[others[m]], 2), 2) : at
    fields = output_fields(sim)

    open(path, "w") do f
        print(f, "# x")
        for (name, nc, _) in fields
            if nc == 1
                print(f, " ", name)
            else
                for c in 1:nc
                    print(f, " ", name, "_", ("x", "y", "z")[c])
                end
            end
        end
        println(f)
        for i in 1:g.n[dir]
            sub = ntuple(3) do d
                d == dir ? i : (d == others[1] ? idx[1] : idx[2])
            end
            x = g.xlo[dir] + (i - 0.5) * g.dx[dir]
            @printf(f, "%.10g", x)
            for (_, nc, data) in fields
                for c in 1:nc
                    @printf(f, " %.10g", data[c, sub[1], sub[2], sub[3]])
                end
            end
            println(f)
        end
    end
    return path
end

# --- checkpoints -------------------------------------------------------------

const _CHECKPOINT_MAGIC = b"HYDRO3D\x01"

"""
    save_checkpoint(sim, path) -> String

Write a raw binary checkpoint (grid metadata plus the full conserved state,
ghost cells included) that [`load_checkpoint!`](@ref) can restore. The file
extension `.h3d` is added if absent.
"""
function save_checkpoint(sim::Simulation{T}, path::AbstractString) where {T}
    fn = endswith(path, ".h3d") ? String(path) : path * ".h3d"
    g = sim.grid
    open(fn, "w") do f
        write(f, _CHECKPOINT_MAGIC)
        write(f, Int64(sizeof(T)))
        for d in 1:3
            write(f, Int64(g.n[d]))
        end
        for d in 1:3
            write(f, Int64(g.ng[d]))
        end
        write(f, Float64(sim.t), Float64(sim.dt), Int64(sim.step))
        write(f, Int64(has_radiation(sim) ? 1 : 0))
        write(f, sim.U)
        has_radiation(sim) && write(f, sim.R)
    end
    return fn
end

"""
    load_checkpoint!(sim, path) -> sim

Restore a simulation from a checkpoint written by [`save_checkpoint`](@ref).
The grid, floating-point precision, and presence of radiation must match those
of `sim`.
"""
function load_checkpoint!(sim::Simulation{T}, path::AbstractString) where {T}
    g = sim.grid
    open(path, "r") do f
        magic = read(f, length(_CHECKPOINT_MAGIC))
        magic == _CHECKPOINT_MAGIC || error("$path is not a Hydro3D checkpoint")
        read(f, Int64) == sizeof(T) || error("checkpoint precision does not match the simulation")
        for d in 1:3
            read(f, Int64) == g.n[d] || error("checkpoint grid size does not match the simulation")
        end
        for d in 1:3
            read(f, Int64) == g.ng[d] || error("checkpoint ghost width does not match the simulation")
        end
        sim.t = T(read(f, Float64))
        sim.dt = T(read(f, Float64))
        sim.step = Int(read(f, Int64))
        hasrad = read(f, Int64) == 1
        hasrad == has_radiation(sim) ||
            error("checkpoint radiation setting does not match the simulation")
        read!(f, sim.U)
        hasrad && read!(f, sim.R)
    end
    return sim
end
