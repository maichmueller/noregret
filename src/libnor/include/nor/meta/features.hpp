#ifndef NOR_META_FEATURES_HPP
#define NOR_META_FEATURES_HPP

// C++26 static reflection (P2996) is a project prerequisite. The supported
// compiler contract is GCC 16.1 or newer, C++26 mode, and -freflection.
#if ! defined(__GNUC__) || defined(__clang__) || (__GNUC__ < 16) \
   || (__GNUC__ == 16 && __GNUC_MINOR__ < 1)
   #error "NOR requires GCC 16.1 or newer with C++26 reflection enabled via -freflection"
#endif

#if ! defined(__cplusplus) || __cplusplus < 202400L
   #error "NOR requires GCC 16.1 or newer with C++26 reflection enabled via -freflection"
#endif

#if ! defined(__has_include) || ! __has_include(<meta>)
   #error "NOR requires the C++26 <meta> header from GCC 16.1 or newer"
#endif

#include <meta>

#if ! defined(__cpp_impl_reflection) || ! defined(__cpp_lib_reflection)
   #error "NOR requires GCC 16.1 or newer with C++26 reflection enabled via -freflection"
#endif

#endif  // NOR_META_FEATURES_HPP
