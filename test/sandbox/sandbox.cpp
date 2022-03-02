

#include "sandbox.hpp"

#include <cppitertools/enumerate.hpp>
#include <cppitertools/reversed.hpp>
#include <iomanip>
#include <iostream>
#include <range/v3/all.hpp>
#include <string>
#include <utility>

#include "nor/nor.hpp"
#include "stratego/Logic.hpp"

/**
 * @brief type_trait to check if functior F can be invoked with each arg
 * @tparam F
 * @tparam Ret
 * @tparam HeadArg
 * @tparam TailArg
 */
template < typename F, typename Ret, typename HeadArg, typename... TailArg >
struct invocable_with_each {
   inline constexpr static bool value = invocable_with_each< F, Ret, HeadArg >::value
                                        && invocable_with_each< F, Ret, TailArg... >::value;
};

template < typename F, typename Ret, typename HeadArg, typename TailArg >
struct invocable_with_each< F, Ret, HeadArg, TailArg > {
   inline constexpr static bool value = invocable_with_each< F, Ret, HeadArg >::value
                                        && invocable_with_each< F, Ret, TailArg >::value;
};

template < typename F, typename Ret, typename Arg >
struct invocable_with_each< F, Ret, Arg > {
  private:
   inline constexpr static bool is_invocable = std::invocable< F, Arg >;

   constexpr static bool correct_return_result()
   {
      if constexpr(is_invocable) {
         return std::is_same_v< std::invoke_result_t< F, Arg >, Ret >;
      } else {
         return false;
      }
   }

  public:
   inline constexpr static bool value = is_invocable && correct_return_result();
};

template < typename F, typename Ret, typename... Args >
inline constexpr bool invocable_with_each_v = invocable_with_each< F, Ret, Args... >::value;

template < typename T, typename Ret, typename... Args >
concept invoc = invocable_with_each_v< T, Ret, Args... >;

class T {
  public:
   void operator()(int);
   void operator()(double);
   //   void operator()(std::string);

   void g(int);
   void g(double);
   //   void g(std::string);
};

void f(int);

template <
   typename Map,
   typename KeyType = typename Map::key_type,
   typename MappedType = typename Map::mapped_type >
concept map = requires(Map m, KeyType key)
{
   {
      m.at(key)
      } -> std::same_as< MappedType& >;
};

template < typename T, typename OutputT, typename InputT >
concept getitem = requires(T&& t, InputT inp)
{
   /// getitem method for input type returning an output type
   {
      t[inp]
      } -> std::same_as< OutputT >;
};

class Lambda {
  public:
   inline /*constexpr */ void operator()() const {}

  private:
   std::shared_ptr< nor::games::stratego::Environment > env;

  public:
   inline Lambda& operator=(const Lambda&) = delete;
   Lambda(const std::shared_ptr< nor::games::stratego::Environment >& _env) : env{_env} {}
};

int main(int argc, char** argv)
{
   //   std::cout << btos(
   //      std::invocable<
   //         decltype([]< typename U >(U u) requires(std::invocable< T, U >) { std::declval< T
   //         >(u); }), long >)
   //             << std::endl;
   //   //   std::cout << btos(std::invocable< decltype([](int a) {}), long >) << std::endl;
   //   std::cout << btos(invoc< T, void, int, double, long >) << std::endl;
   //   std::cout << btos(invoc< T, void, int, double, std::string >) << std::endl;
   //   //   std::cout << btos(
   //   //      invoc< decltype([t = std::declval<T>()](auto a) { t.g(a); }), void, int, double,
   //   //      std::string >)
   //   //             << std::endl;
   //   std::cout << btos(invoc< decltype(&f), void, int >) << std::endl;
   //   std::cout << btos(invoc< decltype(&f), void, int >) << std::endl;
   //   std::cout << btos(invoc< decltype(&f), void, long >) << std::endl;
   //   std::cout << S_m_v<int, double, int> << std::endl;
   //   std::cout << btos(nor::concepts::map< std::map< int, T > >);

   using namespace nor::games::stratego;
   auto env = std::make_shared< Environment >(std::make_unique< Logic >());

   using ActionPolicy = std::unordered_map< Action, double >;

   nor::UniformPolicy def_policy{
      [env = env](const InfoState& istate) { return env->actions(istate); },
      nor::Hint< InfoState, Action, ActionPolicy >{}};
   std::cout << nor::utils::btos(std::is_move_assignable_v< decltype([env = env]() {}) >)
             << std::endl;  //
   std::cout << nor::utils::btos(std::is_move_assignable_v< Lambda >);  //
   //   std::map<int,T>{}.at();
}