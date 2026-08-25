#pragma once

// Minimal test-only stand-in for the real core/math/butterworth_filter.h (not
// present in this checkout). acceleration_matrix.h includes it but does not
// currently use any symbol from it, so an empty header is enough to satisfy
// the include. Swap in the real header from the production monorepo for
// anything else.
