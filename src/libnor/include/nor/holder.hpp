#ifndef NOR_HOLDER_HPP
#define NOR_HOLDER_HPP

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/game_defs.hpp"
#include "nor/rm/tag.hpp"
#include "nor/utils/utils.hpp"

namespace nor {

template < template < typename > class DerivedHolder, typename Type >
struct BaseHolder {
   using derived_wrapper_type = DerivedHolder< Type >;
   using underlying_type = std::remove_cvref_t< Type >;
   static constexpr bool is_polymorphic = std::is_polymorphic_v< underlying_type >;
   // convenience def
   static constexpr bool is_not_polymorphic = not is_polymorphic;
   using holder_type = std::
      conditional_t< is_polymorphic, sptr< underlying_type >, underlying_type >;

  protected:
   /// internal constructor to actually allocate the member
   template < typename... Ts >
   BaseHolder(tag::internal_construct, Ts&&... args)
       : m_member(_init_member(std::forward< Ts >(args)...))
   {
   }

  public:
   template < typename T1, typename... Ts >
      requires common::is_none_v<
         std::remove_cvref_t< T1 >,  // remove_cvref to avoid checking each ref-case individually
         tag::internal_construct,  // don't recurse back from internal constructors or self
         BaseHolder,  // don't steal the copy/move constructor calls
         DerivedHolder< underlying_type >  // (std::remove_cvref ensures both)
         >
   explicit BaseHolder(T1&& t1, Ts&&... args)
       : BaseHolder(tag::internal_construct{}, std::forward< T1 >(t1), std::forward< Ts >(args)...)
   {
   }

   BaseHolder()
      requires std::is_default_constructible_v< underlying_type >
       : BaseHolder(tag::internal_construct{})
   {
   }
   /// implicit converion operators so that the contained type will be passable to functions that
   /// use the underlying type (similar to reference_wrapper's behaviour)
   constexpr operator const underlying_type&() const noexcept { return get(); }
   constexpr operator underlying_type&() noexcept { return get(); }
   /// implicit converion operators so that the passable to functions that use a shared_ptr of the
   /// underlying type
   constexpr operator const sptr< holder_type >&() const noexcept
      requires is_polymorphic
   {
      return m_member;
   }
   constexpr operator sptr< holder_type >&() noexcept
      requires is_polymorphic
   {
      return m_member;
   }

   [[nodiscard]] const underlying_type& get() const
   {
      if constexpr(is_polymorphic) {
         return *m_member;
      } else {
         return m_member;
      }
   }
   [[nodiscard]] underlying_type& get()
   {
      // clang-tidy complains, but since the member function calling it is non-const, the object
      // itself is non-const. Thus, casting away the const is fine.
      return const_cast< underlying_type& >(std::as_const(*this).get());
   }

   /// equals always compares by value of the underlying type
   bool equals(const BaseHolder& other) const
      requires(std::equality_comparable< underlying_type >)
   {
      return get() == other.get();
   }
   /// equality operator compares either by value or by pointer value depending on the holder type
   bool operator==(const BaseHolder& other) const { return m_member == other.m_member; }

   [[nodiscard]] derived_wrapper_type copy() const
   {
      // copy_tag to avoid the implicit copy constructor
      return derived_wrapper_type{tag::internal_construct{}, get()};
   }

   [[nodiscard]] std::string to_string() const
      requires requires(underlying_type obj) { obj.to_string(); }
   {
      return get().to_string();
   }

   const BaseHolder* operator->() const { return &get(); }
   BaseHolder* operator->() { return &get(); }

  private:
   holder_type m_member;

   static constexpr bool _nothrow_default_constructible =
      is_polymorphic or (is_not_polymorphic and std::is_nothrow_constructible_v< underlying_type >);

   static constexpr bool _nothrow_constructible_from_lvalue =
      is_polymorphic
      or (is_not_polymorphic and std::is_nothrow_copy_constructible_v< underlying_type >);

   template < typename FirstArg, typename... Args >
      requires(
         not (
            sizeof...(Args) == 0
            and common::is_any_v< //
               std::remove_cvref_t< FirstArg >,  //
               underlying_type*,  //
               sptr< underlying_type >  //
               >
         )
         and not std::same_as<std::remove_cvref_t< FirstArg >, tag::internal_construct>
      )
   [[nodiscard]] holder_type _init_member(FirstArg&& first_arg, Args&&... args) const
      noexcept(noexcept(
         _init_member_impl(std::forward< FirstArg >(first_arg), std::forward< Args >(args)...)
      ))
   {
      return _init_member_impl(std::forward< FirstArg >(first_arg), std::forward< Args >(args)...);
   }

   holder_type _init_member() const noexcept(_nothrow_default_constructible)
      requires(
         is_polymorphic
         or (is_not_polymorphic and std::is_default_constructible_v< underlying_type >)
      )
   {
      // we either return a nullpointer in the polymorphic case or a default constructed value of
      // the underlying type
      return holder_type{};
   }

   template < typename... Ts >
      requires(sizeof...(Ts) > 0)
   holder_type _init_member_impl(Ts&&... ts) const
      noexcept(is_not_polymorphic and noexcept(underlying_type{std::forward< Ts >(ts)...}))
   {
      if constexpr(is_polymorphic) {
         return std::make_shared< underlying_type >(std::forward< Ts >(ts)...);
      } else {
         return underlying_type{std::forward< Ts >(ts)...};
      }
   };

   template < typename T >
      requires common::
         is_any_v< std::remove_cvref_t< T >, underlying_type*, sptr< underlying_type > >
      holder_type _init_member_impl(std::conditional_t< is_polymorphic, T, const T& > ptr) const
      noexcept(_nothrow_constructible_from_lvalue)
   {
      if constexpr(is_polymorphic) {
         // if we are only given a pointer to the underlying type then we assume the wrapper
         // is supposed to take ownership of it and pass it to the smart pointer's constructor
         return holder_type{std::move(ptr)};
      } else {
         // otherwise we simply copy the contents
         return underlying_type{*ptr};
      }
   }
};

template < typename Action >
struct ActionHolder: public BaseHolder< ActionHolder, Action > {
   using base = BaseHolder< ActionHolder, Action >;
   using base::base;
};

template < typename ChanceOutcome >
struct ChanceOutcomeHolder: public BaseHolder< ChanceOutcomeHolder, ChanceOutcome > {
   using base = BaseHolder< ChanceOutcomeHolder, ChanceOutcome >;
   using base::base;
};

template < typename Observation >
struct ObservationHolder: public BaseHolder< ObservationHolder, Observation > {
   using base = BaseHolder< ObservationHolder, Observation >;
   using base::base;
};

template < typename Worldstate >
struct WorldstateHolder: public BaseHolder< WorldstateHolder, Worldstate > {
   using base = BaseHolder< WorldstateHolder, Worldstate >;
   using base::base;
};

template < typename Infostate >
struct InfostateHolder: public BaseHolder< InfostateHolder, Infostate > {
   using base = BaseHolder< InfostateHolder, Infostate >;
   using base::base;
   using base::get;
   using observation_type = typename fosg_auto_traits< Infostate >::observation_type;

   [[nodiscard]] size_t size() const { return get().size(); }

   const auto& update(const observation_type& public_obs, const observation_type& private_obs)
   {
      return get().update(public_obs, private_obs);
   }

   auto& operator[](auto index) { return get()[size_t(index)]; }

   const auto& operator[](auto index) const { return get()[size_t(index)]; }

   [[nodiscard]] Player player() const { return get().player(); }
};

template < typename Publicstate >
struct PublicstateHolder: public BaseHolder< PublicstateHolder, Publicstate > {
   using base = BaseHolder< PublicstateHolder, Publicstate >;
   using base::base;
   using base::get;
   using observation_type = typename fosg_auto_traits< Publicstate >::observation_type;

   [[nodiscard]] size_t size() const { return get().size(); }

   const auto& update(const observation_type& obs) { return get().update(obs); }

   auto& operator[](auto index) { return get()[size_t(index)]; }

   const auto& operator[](auto index) const { return get()[size_t(index)]; }
};

}  // namespace nor
namespace std {

template < typename T >
   requires common::logical_or_v<
      common::is_specialization< std::remove_cvref_t< T >, nor::ActionHolder >,
      common::is_specialization< std::remove_cvref_t< T >, nor::ObservationHolder >,
      common::is_specialization< std::remove_cvref_t< T >, nor::InfostateHolder >,
      common::is_specialization< std::remove_cvref_t< T >, nor::PublicstateHolder > >
struct hash< T > {
   size_t operator()(const T& t) const
   {
      constexpr auto underlying_hasher = std::hash< std::conditional_t<
         std::remove_cvref_t< T >::is_polymorphic,
         typename std::remove_cvref_t< T >::underlying_type::element_type,
         typename std::remove_cvref_t< T >::underlying_type > >{};

      return underlying_hasher(t.get());
   }
};

}  // namespace std

#endif  // NOR_HOLDER_HPP
