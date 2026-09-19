# Bitwise Operations: A Practical Reference

## The operators

| Operator | Name | What it does |
|---|---|---|
| `&` | AND | 1 only where **both** bits are 1 |
| `\|` | OR | 1 where **either** bit is 1 |
| `^` | XOR | 1 where the bits **differ** |
| `~` | NOT | flips every bit (1↔0) |
| `<<` | Left shift | shifts bits left, filling with 0s on the right |
| `>>` | Right shift | shifts bits right (see note on signed types below) |

```text
  1010   (10)        1010          1010          1010
& 0110    (6)       | 0110        ^ 0110        ~ 1010
------              ------        ------        ------
  0010    (2)         1110  (14)    1100  (12)    0101  (assuming 4-bit width)
```

A note on `>>` with signed integers: right-shifting an *unsigned* value always fills with 0s (**logical shift**). Right-shifting a *signed, negative* value is implementation-defined in older C/C++ standards but, in practice on virtually every real compiler/platform, fills with the sign bit (**arithmetic shift**) — as of C++20 this is standardized. When you're using shifts for bit manipulation rather than arithmetic, prefer `unsigned` types to sidestep any ambiguity entirely.

## Bit numbering
Bits are conventionally numbered from the **least significant bit (LSB)**, bit 0, up to the **most significant bit (MSB)**. For a 32-bit `unsigned int`, bit 0 is the `1`s place and bit 31 is the sign/high bit:

```text
bit:    31 30 ... 2  1  0
value:   0  0  ...  1  0  1    (this is decimal 5: bits 0 and 2 are set)
```

`1 << n` is therefore a very common idiom for "a value with only bit `n` set" — it comes up constantly in the idioms below.

## Setting, clearing, toggling, and checking a bit
These four idioms account for a large fraction of the bitwise code you'll see in real projects — driver code, flag registers, permission systems, etc:

```c
x |=  (1 << n);   // set bit n to 1
x &= ~(1 << n);   // clear bit n to 0
x ^=  (1 << n);   // toggle bit n
if (x & (1 << n)) { /* bit n is set */ }
```

- **Set**: OR-ing with a mask that has only bit `n` set forces that bit to 1 without disturbing any other bit (OR-ing with 0 leaves a bit unchanged).
- **Clear**: AND-ing with the *complement* of that mask forces bit `n` to 0 while leaving every other bit unchanged (AND-ing with 1 leaves a bit unchanged).
- **Toggle**: XOR-ing with the mask flips exactly that bit (XOR-ing with 0 leaves a bit unchanged, XOR-ing with 1 flips it).
- **Check**: AND-ing with the mask leaves only that bit's value, which is nonzero (truthy) exactly when the bit was set.

## Specifying and computing sizes
This is one of the most common places bitwise operators show up in professional code, precisely because computer memory is inherently binary — every power-of-two byte count has an exact, unambiguous bit-shift representation.

### Byte-size constants
Rather than writing out `1024`, `1048576`, etc., it's idiomatic to define size constants as shifts, which makes the power-of-two relationship self-documenting:

```c
#define KB (1u << 10)   // 1,024
#define MB (1u << 20)   // 1,048,576
#define GB (1u << 30)   // 1,073,741,824

size_t bufferSize = 4 * MB;         // clearly "4 megabytes"
```

### Bit width of a type
`sizeof(x)` gives size in **bytes**; multiplying by 8 (or shifting `<< 3`) gives the size in **bits** — useful when writing generic bit-manipulation code that needs to work across `int`, `long`, etc.:

```c
#define BITS(x) (sizeof(x) * 8)
```

### Checking if a value is a power of two
A power of two has exactly one bit set. `n & (n - 1)` clears the lowest set bit of `n`; if `n` was a power of two, that was its *only* set bit, so the result is 0:

```c
bool isPowerOfTwo(unsigned n) {
    return n > 0 && (n & (n - 1)) == 0;
}
```

### Rounding a size up to the next power of two
This comes up constantly when sizing buffers, hash tables, or GPU launch configurations, where a power-of-two size simplifies later masking/indexing arithmetic. The classic "bit-smear" technique: propagate the highest set bit rightward across every lower bit, so the whole value becomes all 1s up to that point, then add 1:

```c
unsigned nextPowerOfTwo(unsigned n) {
    n--;            // handles the case where n is already a power of two
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;   // repeat doubling the shift until it covers the full width
    n++;
    return n;
}
```
E.g., `nextPowerOfTwo(100)` → 128. Each `n |= n >> k` doubles how far the highest bit has "smeared," so after `log2(width)` steps every bit below the original MSB is forced to 1; incrementing then carries all the way up to the next power of two.

### Rounding a size up to a multiple of an alignment boundary
Memory allocators and buffer-management code frequently need to round a requested size up to the nearest multiple of some power-of-two alignment (e.g., a cache line, a page, or a GPU memory transaction size). If `align` is a power of two, `align - 1` is a mask of all the low bits below it, and the following avoids a division/modulo entirely:

```c
size_t alignUp(size_t size, size_t align) {
    return (size + align - 1) & ~(align - 1);
}
```
E.g., `alignUp(37, 16)` → 48. Adding `align - 1` "pushes" any non-zero remainder into the next multiple, and the `& ~(align - 1)` then masks off those low bits, snapping the result back down to the boundary it just crossed.

## Fast arithmetic with power-of-two operands
Because shifting a binary number left/right by `k` positions is exactly multiplying/dividing by $2^k$, compilers routinely translate multiplication/division/modulo by a power of two into shifts and masks automatically — but you'll still see it written explicitly in performance-sensitive or embedded code:

```c
x << k     // x * 2^k        (for unsigned x, or non-overflowing signed x)
x >> k     // x / 2^k        (for unsigned x; arithmetic-shifts a signed x toward negative infinity, not truncating like /)
x & (n - 1) // x % n, only valid when n is a power of two
```
The last one is especially common: `x % n` normally involves a division, which is one of the slowest basic arithmetic operations; `x & (n - 1)` computes the exact same result in a single, fast AND when `n` is a power of two.

## Bit flags and option sets
Bitwise operators are the standard way to pack many independent boolean options into a single integer — you'll see this constantly in library and OS-level APIs (POSIX file `open()` flags, GPU API creation flags, GUI window-style flags, etc.):

```c
enum Flags {
    FLAG_READ    = 1 << 0,  // 0b0001
    FLAG_WRITE   = 1 << 1,  // 0b0010
    FLAG_EXECUTE = 1 << 2,  // 0b0100
    FLAG_HIDDEN  = 1 << 3,  // 0b1000
};

unsigned perms = FLAG_READ | FLAG_WRITE;      // combine flags with OR
perms |= FLAG_EXECUTE;                        // add a flag
perms &= ~FLAG_WRITE;                         // remove a flag
bool canRead = (perms & FLAG_READ) != 0;      // test a flag
```
Writing each flag as `1 << n` (rather than `1, 2, 3, 4...`) makes it visually obvious that each one occupies its own, non-overlapping bit — which is what lets them be safely combined with `|` without stepping on each other.

## Packing and unpacking values
Bitwise shifts and masks are also the standard way to pack several smaller values into one wider integer, or pull them back apart — common in graphics, networking, and file formats:

```c
// pack 4 bytes (R,G,B,A) into one 32-bit color
uint32_t packRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return (r << 24) | (g << 16) | (b << 8) | a;
}

// unpack them back out
uint8_t r = (color >> 24) & 0xFF;
uint8_t g = (color >> 16) & 0xFF;
uint8_t b = (color >>  8) & 0xFF;
uint8_t a =  color         & 0xFF;
```
Each channel is shifted into its own 8-bit "slot," and `& 0xFF` on unpacking discards everything outside the 8 bits being extracted. The same shift-and-mask pattern underlies IPv4 address/subnet-mask manipulation, reading multi-byte values from a binary file format, and CPU/GPU instruction encoding.

## Other common idioms seen in professional code
```c
n << 1;                         // multiply by 2
n >> 1;                         // divide by 2 (fast halving)
(n & 1) == 1;                   // check if n is odd
a ^= b; b ^= a; a ^= b;         // swap two integers without a temporary variable
x & (-x);                       // isolate the lowest set bit of x
x & (x - 1);                    // clear the lowest set bit of x (also the power-of-two check above)
(x ^ y) >= 0;                   // check whether x and y have the same sign
```
These show up often enough in real codebases (especially systems/embedded/performance-critical code) that recognizing them at a glance is worth the small amount of memorization — though most of them (aside from masking/flags) are more often *encountered while reading* code than something you'd reach for first when *writing* new code, since a clearer, more literal equivalent (`n % 2 == 0`, a `std::swap`, etc.) is usually easier to read and just as fast once compiled.

## Where this shows up in CUDA/GPU code specifically
Tying back to the earlier notes in this set — bitwise operations appear constantly in CUDA kernels because warp size (32) and typical block/data sizes are powers of two:
```c
unsigned laneId = threadIdx.x & 31;    // lane index within the warp (0-31), instead of threadIdx.x % 32
unsigned warpId = threadIdx.x >> 5;    // which warp this thread belongs to, instead of threadIdx.x / 32
```
These are exactly the fast-modulo/fast-divide idioms above, applied to `warpSize = 32 = 2^5`.
- The reduction kernels covered earlier use `stride >>= 1` (equivalent to `stride /= 2`) to halve the stride each round — a direct application of the "divide by a power of two via shift" idiom.
- **Vote/ballot intrinsics** like `__ballot_sync()` return a 32-bit mask with one bit per lane in the warp, and code that consumes that mask typically uses exactly the bit-test/bit-count idioms above (e.g., `__popc()` to count how many lanes had a condition true).
- **Bank/address arithmetic** (from the shared-memory notes): `bank = address & (numBanks - 1)` is the same power-of-two-modulo idiom used to compute which shared-memory bank an address falls into, since the bank count is itself always a power of two (32).

## Quick reference

| Goal | Idiom |
|---|---|
| Set bit `n` | `x \|= (1 << n)` |
| Clear bit `n` | `x &= ~(1 << n)` |
| Toggle bit `n` | `x ^= (1 << n)` |
| Test bit `n` | `x & (1 << n)` |
| `n`-kilobyte/megabyte/gigabyte constant | `1u << 10` / `1u << 20` / `1u << 30` |
| Bit width of a type | `sizeof(x) * 8` |
| Is `n` a power of two? | `n > 0 && (n & (n - 1)) == 0` |
| Round up to next power of two | "smear-then-increment" (see above) |
| Round `size` up to a power-of-two `align` | `(size + align - 1) & ~(align - 1)` |
| Multiply/divide by $2^k$ | `x << k` / `x >> k` |
| Modulo a power of two `n` | `x & (n - 1)` |
| Combine / remove / test a flag | `\|=` / `&= ~` / `&` |
| Isolate lowest set bit | `x & (-x)` |
| Clear lowest set bit | `x & (x - 1)` |