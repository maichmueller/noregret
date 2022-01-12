
#pragma once

#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "aze/types.h"

namespace aze::utils {

/**
 * @brief Creates and returns a new random number generator from a potential seed.
 * @param seed the seed for the Mersenne Twister algorithm.
 * @return The Mersenne Twister RNG object
 */
auto create_rng(auto seed = std::nullopt)
{
   return std::mt19937_64{seed.has_value() ? seed.value() : std::random_device{}()};
}

inline std::string repeat(std::string str, const std::size_t n)
{
   if(n == 0) {
      str.clear();
      str.shrink_to_fit();
      return str;
   } else if(n == 1 || str.empty()) {
      return str;
   }
   const auto period = str.size();
   if(period == 1) {
      str.append(n - 1, str.front());
      return str;
   }
   str.reserve(period * n);
   std::size_t m{2};
   for(; m < n; m *= 2)
      str += str;
   str.append(str.c_str(), (n - (m / 2)) * period);
   return str;
}

inline std::string center(const std::string& str, int width, const char* fillchar)
{
   int len = str.length();
   if(width < len) {
      return str;
   }

   int diff = width - len;
   int pad1 = diff / 2;
   int pad2 = diff - pad1;
   return std::string(pad2, *fillchar) + str + std::string(pad1, *fillchar);
}

inline std::string operator*(std::string str, std::size_t n)
{
   return repeat(std::move(str), n);
}

template < typename BoardType, typename PieceType >
std::string
board_str_rep(const BoardType& board, bool flip_board = false, bool hide_unknowns = false)
{
   int H_SIZE_PER_PIECE = 9;
   int V_SIZE_PER_PIECE = 3;
   // the space needed to assign row indices to the rows and to add a splitting
   // bar "|"
   int row_ind_space = 4;

   int mid = V_SIZE_PER_PIECE / 2;

   int dim = board.get_board_len();

   if(dim != 5 && dim != 7 && dim != 10)
      throw std::invalid_argument("Board dimension not supported.");

   // piece string lambda function that returns a str of the sort
   // "-1 \n
   // 10.1 \n
   //   1"
   auto create_piece_str = [&H_SIZE_PER_PIECE, &mid, &flip_board, &hide_unknowns](
                              const PieceType& piece, int line) {
      if(piece.is_null())
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');
      std::string reset = "\x1B[0m";
      std::string color = "\x1B[44m";  // blue by default (for team 1)
      if(piece.get_team() == 99)
         return "\x1B[30;47m" + center("", H_SIZE_PER_PIECE, " ") + "\x1B[0m";
      else if(piece.get_team(flip_board) == 0) {
         color = "\x1B[41m";  // background red, text "white"
      }
      if(line == mid - 1) {
         // hidden info line
         std::string h = piece.get_flag_hidden() ? "?" : " ";
         // return color + center(h, H_SIZE_PER_PIECE, " ") + reset;
         return color + center(h, H_SIZE_PER_PIECE, " ") + reset;
      } else if(line == mid) {
         // type and version info line
         if(hide_unknowns && piece.get_flag_hidden() && piece.get_team(flip_board)) {
            return color + std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ') + reset;
         }
         //                std::cout << "PieceType: type " << piece.get_token() <<
         //                "." << piece.get_version() << " at (" <<
         //                                                                                                          piece.get_position()[0] << ", " << piece.get_position()[1] <<") \n";
         return color
                + center(
                   std::to_string(piece.get_type()) + '.' + std::to_string(piece.get_version()),
                   H_SIZE_PER_PIECE,
                   " ")
                + reset;
      } else if(line == mid + 1)
         // m_team info line
         // return color + center(std::to_string(piece.get_team(flip_board)),
         // H_SIZE_PER_PIECE, " ") + reset;
         return color + center("", H_SIZE_PER_PIECE, " ") + reset;
      else
         // empty line
         return std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ');
   };

   std::stringstream board_print;
   board_print << "\n";
   // column width for the row index plus vertical dash
   board_print << std::string(static_cast< unsigned long >(row_ind_space), ' ');
   // print the column index rows
   for(int i = 0; i < dim; ++i) {
      board_print << center(std::to_string(i), H_SIZE_PER_PIECE + 1, " ");
   }
   board_print << "\n";

   std::string init_space = std::string(static_cast< unsigned long >(row_ind_space), ' ');
   std::string h_border = std::string(
      static_cast< unsigned long >(dim * (H_SIZE_PER_PIECE + 1)), '-');

   board_print << init_space << h_border << "\n";
   std::string init = board_print.str();
   sptr< PieceType > curr_piece;

   // row means row of the board. not actual rows of console output.
   for(int row = 0; row < dim; ++row) {
      // per piece we have V_SIZE_PER_PIECE many lines to fill consecutively.
      // Iterate over every column and append the new segment to the right line.
      std::vector< std::stringstream > line_streams(static_cast< unsigned int >(V_SIZE_PER_PIECE));

      for(int col = 0; col < dim; ++col) {
         if(flip_board) {
            curr_piece = board[{dim - 1 - row, dim - 1 - col}];
         } else
            curr_piece = board[{row, col}];

         for(int i = 0; i < V_SIZE_PER_PIECE; ++i) {
            std::stringstream curr_stream;

            if(i == mid - 1 || i == mid + 1) {
               if(col == 0) {
                  curr_stream << std::string(static_cast< unsigned long >(row_ind_space), ' ');
               }
               curr_stream << "|" << create_piece_str(*curr_piece, i);
            } else if(i == mid) {
               if(col == 0) {
                  if(row < 10)
                     curr_stream << " " << row;
                  else
                     curr_stream << row;

                  curr_stream << std::string(static_cast< unsigned long >(row_ind_space - 2), ' ')
                              << "|";
               }
               curr_stream << create_piece_str(*curr_piece, i);
               if(col != dim - 1)
                  curr_stream << "|";
            }
            // extend the current line i by the new information
            line_streams[i] << curr_stream.str();
         }
      }
      for(auto& stream : line_streams) {
         board_print << stream.str() << "|\n";
      }
      board_print << init_space << h_border << "\n";
   }

   return board_print.str();
}

template < typename BoardType, typename PieceType >
inline void print_board(const BoardType& board, bool flip_board = false, bool hide_unknowns = false)
{
   std::string output = board_str_rep< BoardType, PieceType >(board, flip_board, hide_unknowns);
   std::cout << output << std::endl;
}

inline std::map< int, unsigned int > counter(const std::vector< int >& vals)
{
   std::map< int, unsigned int > rv;

   for(auto val = vals.begin(); val != vals.end(); ++val) {
      rv[*val]++;
   }

   return rv;
}

template < int N >
struct faculty {
   static constexpr int n_faculty() { return N * faculty< N - 1 >::n_faculty(); }
};

template <>
struct faculty< 0 > {
   static constexpr int n_faculty() { return 1; }
};

template < class first, class second, class... types >
auto min(first f, second s, types... t)
{
   return f < s ? min(f, t...) : min(s, t...);
}

template < class first, class second >
auto min(first f, second s)
{
   return f < s ? f : s;
}

template < typename Tuple, size_t... I >
auto call_min_from_tuple(Tuple& args, std::index_sequence< I... >)
{
   return min(std::get< I >(args)...);
}

};  // namespace aze::utils

#include <tuple>

namespace tuple {

template < typename TT >
struct hash {
   size_t operator()(TT const& tt) const { return std::hash< TT >()(tt); }
};

namespace {
template < class T >
inline void hash_combine(std::size_t& seed, T const& v)
{
   seed ^= tuple::hash< T >()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}  // namespace

namespace {
// Recursive template code derived from Matthieu M.
template < class Tuple, size_t Index = std::tuple_size< Tuple >::value - 1 >
struct HashValueImpl {
   static void apply(size_t& seed, Tuple const& tuple)
   {
      HashValueImpl< Tuple, Index - 1 >::apply(seed, tuple);
      hash_combine(seed, std::get< Index >(tuple));
   }
};

template < class Tuple >
struct HashValueImpl< Tuple, 0 > {
   static void apply(size_t& seed, Tuple const& tuple) { hash_combine(seed, std::get< 0 >(tuple)); }
};
}  // namespace

template < typename... TT >
struct hash< std::tuple< TT... > > {
   size_t operator()(std::tuple< TT... > const& tt) const
   {
      size_t seed = 0;
      HashValueImpl< std::tuple< TT... > >::apply(seed, tt);
      return seed;
   }
};
}  // namespace tuple

namespace {
template < typename T, typename... Ts >
auto head(std::tuple< T, Ts... > const& t)
{
   return std::get< 0 >(t);
}

template < std::size_t... Ns, typename... Ts >
auto tail_impl(std::index_sequence< Ns... > const&, std::tuple< Ts... > const& t)
{
   return std::make_tuple(std::get< Ns + 1u >(t)...);
}

template < typename... Ts >
auto tail(std::tuple< Ts... > const& t)
{
   return tail_impl(std::make_index_sequence< sizeof...(Ts) - 1u >(), t);
}

template < typename TT >
struct eqcomp {
   bool operator()(TT const& tt1, TT const& tt2) const { return ! ((tt1 < tt2) || (tt2 < tt1)); }
};
}  // namespace

namespace std {
template < typename T1, typename... TT >
struct equal_to< std::tuple< T1, TT... > > {
   bool operator()(std::tuple< T1, TT... > const& tuple1, std::tuple< T1, TT... > const& tuple2)
      const
   {
      return eqcomp< T1 >()(std::get< 0 >(tuple1), std::get< 0 >(tuple2))
             && eqcomp< std::tuple< TT... > >()(tail(tuple1), tail(tuple2));
   }
};

template <>
struct hash< tuple< string, int > > {
   size_t operator()(const std::tuple< std::string, int >& s) const
   {
      return std::hash< std::string >()(
         std::get< 0 >(s) + "!@#$%^&*()_" + std::to_string(std::get< 1 >(s)));
   }
};
}  // namespace std
