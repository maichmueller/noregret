#ifndef NOR_META_FEATURES_HPP
#define NOR_META_FEATURES_HPP

// Feature gate for C++26 static reflection (P2996).
//
// NOR_REFLECTION is defined iff the compiler advertises core reflection
// (__cpp_impl_reflection) and the standard library provides the <meta> portion
// (__cpp_lib_reflection). On GCC 16 both are only available in C++26 mode with
// '-freflection', and __cpp_lib_reflection only becomes defined after <meta>
// has been included -- hence the guarded include below.
//
// NOR_FORCE_REFLECTION overrides detection and unconditionally enables the
// reflected code path. It exists to test the reflected path even when the
// feature-test macros misbehave; using it without a reflection-capable setup
// (-std=c++26 -freflection on GCC16) fails to compile by design.

#if defined(__has_include)
   #if __has_include(<meta>)
      #include <meta>
   #endif
#endif

#if defined(NOR_FORCE_REFLECTION)
   #define NOR_REFLECTION 1
#elif defined(__cpp_impl_reflection) && defined(__cpp_lib_reflection)
   #define NOR_REFLECTION 1
#endif

#endif  // NOR_META_FEATURES_HPP
