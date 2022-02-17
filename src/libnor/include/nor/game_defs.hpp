
#ifndef NOR_GAME_DEFS_HPP
#define NOR_GAME_DEFS_HPP

namespace nor {

enum class Player {
   chance = -1,
   alex = 0,
   bob = 1,
   cedric = 2,
   dexter = 3,
   emily = 4,
   florence = 5,
   gustavo = 6,
   henrick = 7,
   ian = 8,
   julia = 9,
   kelvin = 10,
   lea = 11,
   michael = 12,
   norbert = 13,
   oscar = 14,
   pedro = 15,
   quentin = 16,
   rosie = 17,
   sophia = 18,
   tristan = 19,
   ulysses = 20,
   victoria = 21,
   william = 22,
   xavier = 23,
   yeet = 24,
   zoey = 25
};

enum class TurnDynamic {
   sequential = 0,  // sequential actions (only one player acts in a turn)
   simultaneous     // simultaneous actions (every player acts in a turn)
};

enum class Stochasticity {
   deterministic = 0,  // the environment is deterministic
   sample,   // the environment is sampling a random outcome
   choice    // the environment can provide a vector of random outcomes to choose from
};

}

#endif  // NOR_GAME_DEFS_HPP
