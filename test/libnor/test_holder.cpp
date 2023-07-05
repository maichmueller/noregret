
#include <gtest/gtest.h>

#include "dummy_classes.hpp"
#include "nor/holder.hpp"

using namespace nor;

constexpr auto
   very_long_string = "123456789abcdefghijklmnopqrstuvwxzy123456789abcdefghijklmnopqrstuvwxzy";

TEST(Holder, all_class_constants)
{
   using namespace dummy;
   EXPECT_FALSE(ActionHolder< Action >::dynamic_storage);
   EXPECT_FALSE(ActionHolder< Action >::is_polymorphic);
   EXPECT_FALSE(ActionHolder< Action >::force_dynamic_storage);
   EXPECT_TRUE((std::same_as< ActionHolder< Action >::value_type, Action >) );
   EXPECT_TRUE(ActionHolder< PolyActionBase >::dynamic_storage);
   EXPECT_TRUE(ActionHolder< PolyActionBase >::is_polymorphic);
   EXPECT_FALSE(ActionHolder< PolyActionBase >::force_dynamic_storage);
   EXPECT_TRUE((std::same_as< ActionHolder< PolyActionBase >::value_type, sptr< PolyActionBase > >)
   );
   EXPECT_TRUE((ActionHolder< Action, std::true_type >::dynamic_storage));
   EXPECT_FALSE((ActionHolder< Action, std::true_type >::is_polymorphic));
   EXPECT_TRUE((ActionHolder< Action, std::true_type >::force_dynamic_storage));
   EXPECT_TRUE((std::same_as< ActionHolder< Action, std::true_type >::value_type, sptr< Action > >)
   );

   EXPECT_TRUE((std::is_same_v<
                ActionHolder< PolyAction >,
                typename ActionHolder< PolyAction >::derived_holder_type >) );
}

TEST(Holder, construction_brace_initialization_forwarding)
{
   using namespace dummy;
   std::vector< std::string > vec1{"these", "are", "const", "char", "strings"};
   std::vector< std::string > vec2{"hi", "hi", "hi", "hi"};
   BasicHolder< std::vector< std::string > > holder1{"these", "are", "const", "char", "strings"};
   BasicHolder< std::vector< std::string > > holder2(size_t(4), std::string("hi"));

   EXPECT_TRUE(holder1.equals(vec1));
   EXPECT_TRUE(holder2.equals(vec2));
}

TEST(Holder, construction_non_polymorphic)
{
   using namespace dummy;
   Action action{very_long_string, 101};
   // construction from brace init
   ActionHolder< Action > holder1{very_long_string, 101};
   // construction from default init
   ActionHolder< Action > holder2(very_long_string, 101);
   // construction from ref
   ActionHolder< Action > holder3(action);
   // construction from const ref
   ActionHolder< Action > holder4(std::as_const(action));

   // expect all holders to be equal to action value-wise
   EXPECT_EQ(action, holder1.get());
   EXPECT_EQ(action, holder2.get());
   EXPECT_EQ(action, holder3.get());
   EXPECT_EQ(action, holder4.get());

   EXPECT_TRUE(holder1.equals(action));
   EXPECT_TRUE(holder2.equals(action));
   EXPECT_TRUE(holder3.equals(action));
   EXPECT_TRUE(holder4.equals(action));

   // we expect operator== to always test out true due to comparing values
   EXPECT_EQ(holder1, holder2);
   EXPECT_EQ(holder1, holder3);
   EXPECT_EQ(holder1, holder4);
   EXPECT_EQ(holder2, holder3);
   EXPECT_EQ(holder3, holder4);
   // expect all holders to be unequal to action memory-wise
   EXPECT_FALSE(holder1.is(action));
   EXPECT_FALSE(holder2.is(action));
   EXPECT_FALSE(holder3.is(action));
   EXPECT_FALSE(holder4.is(action));
   // and among each other
   EXPECT_FALSE(holder1.is(holder2));
   EXPECT_FALSE(holder1.is(holder3));
   EXPECT_FALSE(holder1.is(holder4));
   EXPECT_FALSE(holder2.is(holder3));
   EXPECT_FALSE(holder2.is(holder4));
   EXPECT_FALSE(holder3.is(holder4));

   // now move the action uptr into the holder
   ActionHolder< Action > holder5(std::move(action));
   // now no longer the same since strings are moved away (unlike primitives)
   EXPECT_FALSE(holder5.equals(action));
   // should have the same value as all other holders though
   EXPECT_TRUE(holder5.equals(holder1));
   EXPECT_TRUE(holder5.equals(holder2));
   EXPECT_TRUE(holder5.equals(holder3));
   EXPECT_TRUE(holder5.equals(holder4));

   EXPECT_EQ(holder5, holder1);
   EXPECT_EQ(holder5, holder2);
   EXPECT_EQ(holder5, holder3);
   EXPECT_EQ(holder5, holder4);
}

TEST(Holder, construction_polymorphic)
{
   using namespace dummy;
   auto action_uptr = std::make_unique< PolyAction >(very_long_string, 101);
   std::shared_ptr< PolyActionBase > action_sptr{
      std::make_unique< PolyAction >(very_long_string, 101)};

   auto& action = *action_uptr;
   // construction from brace derived class raw pointer and taking ownership
   ActionHolder< PolyActionBase > holder1{new PolyAction{very_long_string, 101}};
   // construction from copying shared ptr
   ActionHolder< PolyActionBase > holder2(std::make_unique< PolyAction >(very_long_string, 101));
   // construction from reference to derived class (expecting a clone)
   ActionHolder< PolyActionBase > holder3(*action_uptr);
   // construction from const ref to derived class (expecting a clone)
   ActionHolder< PolyActionBase > holder4(std::as_const(action));

   // use unique ptr first to ensure proper cleanup in case of throw
   auto action2_uptr = std::make_unique< PolyAction >(very_long_string, 101);
   auto raw_ptr = action2_uptr.get();
   {
      // construction from raw pointer, we expect the holder to grab ownership of the memory!
      ActionHolder< PolyActionBase > holder5(raw_ptr);
      // with the deletion of holder5 the pointed to memory by action2_uptr should also be already
      // deleted!
      EXPECT_EQ(raw_ptr, holder5.ptr());
   }
   // manually ensure that the unique pointer no longer attempts to free the already deleted memory
   action2_uptr.release();
   ASSERT_FALSE(bool(action2_uptr));

   EXPECT_EQ(action, holder1.get());
   EXPECT_EQ(action, holder2.get());
   EXPECT_EQ(action, holder3.get());
   EXPECT_EQ(action, holder4.get());

   EXPECT_TRUE(holder1.equals(action));
   EXPECT_TRUE(holder2.equals(action));
   EXPECT_TRUE(holder3.equals(action));
   EXPECT_TRUE(holder4.equals(action));

   // we expect operator== to always test out true due to comparing values
   EXPECT_EQ(holder1, holder2);
   EXPECT_EQ(holder1, holder3);
   EXPECT_EQ(holder1, holder4);
   EXPECT_EQ(holder2, holder3);
   EXPECT_EQ(holder3, holder4);
   // expect all holders to be unequal to action memory-wise
   EXPECT_FALSE(holder1.is(action));
   EXPECT_FALSE(holder2.is(action));
   EXPECT_FALSE(holder3.is(action));
   EXPECT_FALSE(holder4.is(action));
   // and among each other
   EXPECT_FALSE(holder1.is(holder2));
   EXPECT_FALSE(holder1.is(holder3));
   EXPECT_FALSE(holder1.is(holder4));
   EXPECT_FALSE(holder2.is(holder3));
   EXPECT_FALSE(holder2.is(holder4));
   EXPECT_FALSE(holder3.is(holder4));

   // now construct the holder from a rvalue uptr
   ActionHolder< PolyActionBase > holder5(std::move(action_uptr));
   // assert the actionholder actually stole the data in the uptr
   ASSERT_EXIT((holder5.equals(*action_uptr)), ::testing::KilledBySignal(SIGSEGV), ".*");
   // should have the same value as all other holders though
   EXPECT_TRUE(holder5.equals(holder1));
   EXPECT_TRUE(holder5.equals(holder2));
   EXPECT_TRUE(holder5.equals(holder3));
   EXPECT_TRUE(holder5.equals(holder4));

   // now move the action into the holder
   ActionHolder< PolyActionBase > holder6(action_sptr);
   // assert the actionholder actually shares ownership of the data
   EXPECT_EQ(action_sptr.use_count(), 2);
   // should have the same value as all other holders though
   EXPECT_TRUE(holder6.equals(holder1));
   EXPECT_TRUE(holder6.equals(holder2));
   EXPECT_TRUE(holder6.equals(holder3));
   EXPECT_TRUE(holder6.equals(holder4));
   EXPECT_TRUE(holder6.equals(holder5));
}

TEST(Holder, copying)
{
   using namespace dummy;

   auto action_uptr = std::make_unique< PolyAction >(very_long_string, 101);
   // clone the uptr object into the holder
   ActionHolder< PolyActionBase > holder(*action_uptr);

   EXPECT_TRUE(holder.equals(*action_uptr));
   EXPECT_FALSE(holder.is(*action_uptr));
   EXPECT_TRUE(holder.is_not(*action_uptr));

   auto action_sptr = std::make_shared< PolyAction >(very_long_string, 101);
   // copy an existing shared ptr into the holder
   ActionHolder< PolyActionBase > holder2(action_sptr);

   EXPECT_TRUE(holder2.equals(*action_sptr));
   EXPECT_TRUE(holder2.is(*action_sptr));
   EXPECT_FALSE(holder2.is_not(*action_sptr));

   auto holder_copy = holder.copy();
   auto holder_deviating_copy = holder.copy< ActionHolder< PolyActionBase, std::true_type > >();
   EXPECT_TRUE(
      (std::same_as< std::remove_cvref_t< decltype(holder_copy) >, ActionHolder< PolyActionBase > >)
   );
   EXPECT_TRUE((std::same_as<
                std::remove_cvref_t< decltype(holder_deviating_copy) >,
                ActionHolder< PolyActionBase, std::true_type > >) );

   auto holder_cast = dynamic_cast< PolyAction* >(holder.ptr());
   auto nonpoly_holder = ActionHolder< Action >{holder_cast->value(), holder_cast->poly_value()};
   auto nonpoly_holder_copy = nonpoly_holder.copy();
   auto nonpoly_deviating_copy = nonpoly_holder.copy< ActionHolder< Action, std::true_type > >();
   EXPECT_TRUE(
      (std::same_as< std::remove_cvref_t< decltype(nonpoly_holder_copy) >, ActionHolder< Action > >)
   );
   EXPECT_TRUE((std::same_as<
                std::remove_cvref_t< decltype(nonpoly_deviating_copy) >,
                ActionHolder< Action, std::true_type > >) );

   EXPECT_TRUE(nonpoly_holder_copy.equals(nonpoly_holder));
   EXPECT_TRUE(nonpoly_deviating_copy.equals(nonpoly_holder));

   EXPECT_EQ(holder_copy, holder);
   EXPECT_EQ(holder_copy, holder_deviating_copy);

   EXPECT_FALSE(holder_copy.is(holder));
   EXPECT_FALSE(holder_copy.is(holder_deviating_copy));
   EXPECT_TRUE(holder_copy.is_not(holder));
   EXPECT_TRUE(holder_copy.is_not(holder_deviating_copy));
}

TEST(Holder, implicit_conversions)
{
   using namespace dummy;

   EXPECT_TRUE((std::same_as<
                ActionHolder< PolyActionBase >::effective_derived_holder_type,
                ActionHolder< PolyActionBase, std::true_type >::effective_derived_holder_type >) );
   EXPECT_TRUE((std::same_as<
                ActionHolder< PolyActionBase, std::false_type >::effective_derived_holder_type,
                ActionHolder< PolyActionBase, std::true_type >::effective_derived_holder_type >) );

   EXPECT_FALSE((std::same_as<
                 ActionHolder< Action >::effective_derived_holder_type,
                 ActionHolder< Action, std::true_type >::effective_derived_holder_type >) );
   EXPECT_TRUE((std::same_as<
                ActionHolder< Action, std::false_type >::effective_derived_holder_type,
                ActionHolder< Action >::effective_derived_holder_type >) );

   EXPECT_TRUE((std::is_convertible_v< ActionHolder< PolyActionBase >&, PolyActionBase& >) );
   EXPECT_TRUE((std::is_convertible_v< ActionHolder< PolyActionBase >&, const PolyActionBase& >) );
   EXPECT_TRUE(
      (std::is_convertible_v< const ActionHolder< PolyActionBase >&, const PolyActionBase& >)
   );
   EXPECT_FALSE((std::is_convertible_v< const ActionHolder< PolyActionBase >&, PolyActionBase& >) );
}
