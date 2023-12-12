
#ifndef NOR_MACRO_HPP
#define NOR_MACRO_HPP

// adapted from: https://gist.github.com/Redchards/99e56a22bafd1646b21d

#define GCC_COMPILER 1
#define MVSC_COMPILER 2
#define ICC_COMPILER 3
#define BORLAND_COMPILER 4

#define LINUX 1
#define WINDOWS 2
#define MAC 3

#define UNKOWN "unknown"

#if defined(__GNUC__) || defined(__MINGW__) || defined(__clang__)
   #define FUNCTION __PRETTY_FUNCTION__
   #define restrict __restrict__
   #define force_inline __attribute__((always_inline))
   #define likely(x) __builtin_expect((x), 1)
   #define unlikely(x) __builtin_expect((x), 0)
   #if defined(__MINGW__)
      #define COMPILER_NAME "MinGW"
   #else
      #define COMPILER_NAME "gcc"
   #endif
   #define COMPILER GCC_COMPILER
   #if defined(__GNUC_PATCHLEVEL__)
      #define COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
   #else
      #define COMPILER_VERSION (__GNUC__ * 10000 + __GNUC__ * 100)
   #endif
   #if(COMPILER_VERSION < 40500)
      #define DEPRECATED(msg) __attribute__((deprecated))
   #else
      #define DEPRECATED(msg) __attribute__((deprecated(msg)))
   #endif
   #if((COMPILER_VERSION >= 40700) || defined(__clang__))
      #define override override
      #define final final
   #endif
   #define naked __attribute__((naked))
#elif defined(_MSC_VER)
   #define FUNCTION __FUNCSIG__
   #define restrict __declspec(restrict)
   #define force_inline __forceinline
   #define likely(x)
   #define unlikely(x)
   #define COMPILE_NAME "Microsoft compiler"
   #define COMPILER MVSC_COMPILER
   #if((_MSC_VER <= 1200))
      #define COMPILER_VERSION _MSC_VER
   #else
      #define COMPILER_VERSION _MSC_FULL_VER
   #endif
   #if(COMPILER_VERSION >= 1400)
      #define override override
      #define final sealed
   #endif
   #define naked __declspec(naked)
   #define DEPRECATED(msg) __delcspec(deprecated(msg))
#elif defined(__INTEL_COMPILER) || defined(__ICL)
   #define FUNCTION __PRETTY_FUNCTION__
   #define restrict restrict
   #define force_inline  // Could be a way, but I don't know it yet
   #define likely(x)  // No support atm
   #define unlikely(x)
   #define COMPILER_NAME "Intel compiler"
   #define COMPILER ICC_COMPILER
   #define COMPILER_VERSION __INTEL_COMPILER
   #if(COMPILER_VERSION >= 20130000)
      #define override override
      #define final final
   #else
      #define override
      #define final
   #endif
   #define DEPRECATED(msg) __attribute__((deprecated))
   #define naked __attribute__((naked))
#else
   #warning \
      "Unknown compiler, there might be some troubles during compilation, and some special features are disabled. \
			 This may lead to performance problems and/or debugging troubles! Please check list of supported compiler."
   #ifdef(__func__)  // standard since C++ 11 but compiler may not support it
      #define FUNCTION __func__
   #else
      #warning \
         "Compiler is not AINSI C99 standard, can't use __func__. Some debug functions will be disabled"
   #endif
   #define restrict
   #define always_inline
   #define COMPILER_NAME UNKNWON
   #define COMPILER UNKNOWN
   #define COMPILER_VERSION UNKNOWN
   #warning "Compiler not recongnized, results may be unexpected."
   #define DEPRECATED(t)
#endif

#if(! defined override && ! defined final)
   #define override
   #define final
   #warning \
      "Keyword override and final are not defined by your compiler. This may lead to wrong result when using NOR_API. Please change your compiler to a supported compiler!"
#endif

#if defined(__gnu_linux__) || defined(__linux__)
   #define OS_NAME "Linux"
   #define OS LINUX
#elif defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__))
   #define OS_NAME "Mac"
   #define OS MAC
#elif defined(_WIN32)
   #define OS_NAME "Windows"
   #define OS WINDOWS
#else
   #define OS_NAME UNKNOWN
   #define OS UNKNOWN
   #warning "Operating system not recognized. Unexpected results might happen."
#endif

#if(OS == LINUX || OS == MAC) && COMPILER_VERSION >= 4000 \
   && (COMPILER == ICC_COMPILER || COMPILER == GCC_COMPILER)
   #ifndef NOR_EXPORT
      #define NOR_EXPORT __attribute__((visibility("default")))
   #endif
   #ifndef NOR_IMPORT
      #define NOR_IMPORT __attribute__((visibility("default")))
   #endif
   #ifndef NOR_LOCAL
      #define NOR_LOCAL __attribute__((visibility("hidden")))
   #endif
#elif(OS == WINDOWS)
   #define NOR_EXPORT __declspec(dllexport)
   #define NOR_IMPORT __declspec(dllimport)
   #define NOR_LOCAL
#else
   #define NOR_EXPORT
   #define NOR_IMPORT
   #define NOR_LOCALR
#endif

#if ! defined(NOR_BUILD_STATIC) && ! defined(NOR_IMPORT_LIBRARY)
   #define NOR_API NOR_EXPORT
#elif defined(NOR_IMPORT_LIBRARY)
   #define NOR_API NOR_IMPORT
#else
   #define NOR_API
#endif

#if defined(__amd64__) || defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64) \
   || defined(_ia64__) || defined(_M_IA64) || defined(__itanium__)
   #define PLATFORM_X64
#else
   #define PLATFORM_X86
#endif

#if defined(_DEBUG) || ! defined(NDEBUG)
   #define DEBUG true
#else
   #define DEBUG false
#endif

#endif  // NOR_MACRO_HPP
