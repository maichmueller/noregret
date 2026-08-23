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
//
// GCC16 quirk: this header MUST be force-included before any other project
// header in reflection mode (pass '-include <path>/features.hpp' before the
// sources). range/v3/iterator/move_iterators.hpp spells 'meta::_t<...>' inside
// a std::iterator_traits specialization; if std::meta becomes visible after
// that header has been parsed but before instantiation, GCC16's deferred
// template-body check resolves 'meta' to the then-visible std::meta and
// rejects the spelling. Declaring the compatibility alias below up front (in
// every translation unit, before anything else) keeps both worlds working.
// Technically declaring names in namespace std is reserved, but the
// declaration merges with the real std::meta from <meta> and is inert for
// builds that never enter the reflection path below.

// GCC16 + range-v3 0.12.0 quirk: range/v3/iterator/move_iterators.hpp
// specializes std::iterator_traits and spells 'meta::_t<...>' from within
// namespace std. Once <meta> makes 'std::meta' visible, that spelling resolves
// to std::meta instead of the global '::meta' library range-v3 vendors, and
// eager template-body checking rejects it. Declaring a compatible alias in
// std::meta up front keeps both worlds working. Technically declaring names in
// namespace std is reserved, but the declaration merges with the real
// std::meta from <meta> and has no effect on programs that never see this
// header.
#if defined(__cpp_impl_reflection)
namespace std {
namespace meta {
template < typename T >
using _t = typename T::type;
}
}  // namespace std
#endif

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
