
#pragma once

#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <range/v3/all.hpp>
#include <sstream>
#include <string>
#include <utility>

template < typename T >
using uptr = std::unique_ptr< T >;
template < typename T >
using sptr = std::shared_ptr< T >;
template < typename T >
using wptr = std::weak_ptr< T >;

namespace aze::utils {

template < typename T >
requires requires(T t)
{
   {
      std::cout << t
      } -> std::same_as< std::ostream& >;
}
struct VectorPrinter {
   const std::vector< T >& vector;
   std::string delimiter;

   VectorPrinter(const std::vector< T >& vec, std::string delim = ", ")
       : vector(vec), delimiter(std::move(delim))
   {
   }

   friend auto& operator<<(std::ostream& os, const VectorPrinter& printer)
   {
      os << "\n[";
      for(unsigned int i = 0; i < printer.vector.size() - 1; ++i) {
         os << printer.vector[i] << printer.delimiter;
      }
      os << printer.vector.back() << "]";
      return os;
   }
};

template < typename StateType >
class Plotter {
  public:
   virtual ~Plotter() = default;

   virtual void plot(const StateType& state) = 0;
};

template < typename... Ts >
struct Overload: Ts... {
   using Ts::operator()...;
};
template < typename... Ts >
Overload(Ts...) -> Overload< Ts... >;

namespace random {

using RNG = std::mt19937_64;
/**
 * @brief Creates and returns a new random number generator from a potential seed.
 * @param seed the seed for the Mersenne Twister algorithm.
 * @return The Mersenne Twister RNG object
 */
inline auto create_rng()
{
   return RNG{std::random_device{}()};
}
inline auto create_rng(size_t seed)
{
   return RNG{seed};
}
inline auto create_rng(RNG rng)
{
   return rng;
}

template < typename Container>
auto choose(const Container& cont, RNG& rng)
{
   return cont[std::uniform_int_distribution(0ul, cont.size())(rng)];
}
template < typename Container>
auto choose(const Container& cont)
{
   auto dev = std::random_device{};
   return cont[std::uniform_int_distribution(0ul, cont.size())(dev)];
}


}  // namespace random

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
   size_t len = str.length();
   if(std::cmp_less(width, len)) {
      return str;
   }

   int diff = static_cast< int >(static_cast< unsigned long >(width) - len);
   int pad1 = diff / 2;
   int pad2 = diff - pad1;
   return std::string(static_cast< unsigned long >(pad2), *fillchar) + str
          + std::string(static_cast< unsigned long >(pad1), *fillchar);
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
      if(piece.team() == 99)
         return "\x1B[30;47m" + center("", H_SIZE_PER_PIECE, " ") + "\x1B[0m";
      else if(piece.team(flip_board) == 0) {
         color = "\x1B[41m";  // background red, text "white"
      }
      if(line == mid - 1) {
         // hidden info line
         std::string h = piece.flag_hidden() ? "?" : " ";
         // return color + center(h, H_SIZE_PER_PIECE, " ") + reset;
         return color + center(h, H_SIZE_PER_PIECE, " ") + reset;
      } else if(line == mid) {
         // type and version info line
         if(hide_unknowns && piece.flag_hidden() && piece.team(flip_board)) {
            return color + std::string(static_cast< unsigned long >(H_SIZE_PER_PIECE), ' ') + reset;
         }
         //                std::cout << "PieceType: type " << piece.token() <<
         //                "." << piece.get_version() << " at (" <<
         //                                                                                                          piece.get_position()[0] << ", " << piece.position()[1] <<") \n";
         return color
                + center(
                   std::to_string(piece.get_type()) + '.' + std::to_string(piece.get_version()),
                   H_SIZE_PER_PIECE,
                   " ")
                + reset;
      } else if(line == mid + 1)
         // m_team info line
         // return color + center(std::to_string(piece.team(flip_board)),
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

template < typename T >
inline std::map< T, unsigned int > counter(const std::vector< T >& vals)
{
   std::map< T, unsigned int > rv;

   for(auto val = vals.begin(); val != vals.end(); ++val) {
      rv[*val]++;
   }

   return rv;
}

template < typename T, typename Allocator, typename Accessor >
inline auto counter(
   const std::vector< T, Allocator >& vals,
   Accessor acc = [](const auto& iter) { return *iter; })
{
   std::map< T, unsigned int > rv;

   for(auto val_iter = vals.begin(); val_iter != vals.end(); ++val_iter) {
      rv[acc(val_iter)]++;
   }

   return rv;
}

template < typename Container, typename Accessor >
inline auto counter(
   const Container& vals,
   Accessor acc = [](const auto& iter) { return *iter; })
{
   std::map< typename Container::mapped_type, unsigned int > rv;

   for(auto val_iter = vals.begin(); val_iter != vals.end(); ++val_iter) {
      rv[acc(val_iter)]++;
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

template < class T, class... Ts >
struct is_any: ::std::disjunction< ::std::is_same< T, Ts >... > {
};
template < class T, class... Ts >
inline constexpr bool is_any_v = is_any< T, Ts... >::value;

template < class T, class... Ts >
struct is_same: ::std::conjunction< ::std::is_same< T, Ts >... > {
};
template < class T, class... Ts >
inline constexpr bool is_same_v = is_same< T, Ts... >::value;

template < typename Key, typename Value, std::size_t Size >
struct CEMap {
   std::array< std::pair< Key, Value >, Size > data;

   [[nodiscard]] constexpr Value at(const Key& key) const
   {
      const auto itr = std::find_if(
         begin(data), end(data), [&key](const auto& v) { return v.first == key; });
      if(itr != end(data)) {
         return itr->second;
      } else {
         throw std::range_error("Not Found");
      }
   }
};

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
