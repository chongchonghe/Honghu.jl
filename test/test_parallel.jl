# The parallel drivers: each must do exactly what its serial counterpart does,
# whatever the number of threads. Bit-for-bit (`==`) is the requirement, because
# the drivers only change who computes what, never the arithmetic.

@testset "threaded fill and copy" begin
    A = rand(5, 10, 11, 13)
    B = similar(A)
    Honghu.threaded_copyto!(B, A)
    @test B == A
    Honghu.threaded_fill!(A, 0.0)
    @test all(iszero, A)
    # a single slab along the last index, which leaves nothing to split
    C = rand(3, 4, 5, 1)
    D = similar(C)
    Honghu.threaded_copyto!(D, C)
    @test D == C
end
