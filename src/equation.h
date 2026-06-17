#pragma once
// equation.h — Tiny recursive-descent parser that evaluates an integer equation
// string into a GMP big integer (mpz_t).
//
// Grammar (standard precedence, '^' is right-associative):
//   expr    := term   (('+' | '-') term)*
//   term    := factor (('*' | '/' | '%') factor)*
//   factor  := power
//   power   := unary  ('^' power)?
//   unary   := ('+' | '-') unary | primary
//   primary := NUMBER | '(' expr ')'
//
// Supported operators: + - * / % ^ and parentheses. '/' and '%' are integer
// (truncated) division and remainder. Exponents must be non-negative and fit in
// an unsigned long. Numbers may be arbitrarily large decimal literals.
//
// Whitespace is ignored. Throws std::runtime_error on any syntax/semantic error.

#include <gmp.h>
#include <stdexcept>
#include <string>
#include <cctype>

class EquationParser
{
public:
    // Parses `text` and stores the result in `out` (must be mpz_init'd by caller).
    static void eval(const std::string &text, mpz_t out)
    {
        EquationParser p(text);
        p.skip_ws();
        p.parse_expr(out);
        p.skip_ws();
        if (p.pos_ != p.src_.size())
            p.fail("unexpected trailing characters");
    }

private:
    const std::string &src_;
    size_t pos_ = 0;

    explicit EquationParser(const std::string &src) : src_(src) {}

    [[noreturn]] void fail(const char *msg) const
    {
        throw std::runtime_error("equation parse error at column " +
                                 std::to_string(pos_ + 1) + ": " + msg +
                                 "  in \"" + src_ + "\"");
    }

    void skip_ws()
    {
        while (pos_ < src_.size() && std::isspace((unsigned char)src_[pos_]))
            pos_++;
    }

    char peek()
    {
        skip_ws();
        return pos_ < src_.size() ? src_[pos_] : '\0';
    }

    // expr := term (('+'|'-') term)*
    void parse_expr(mpz_t out)
    {
        parse_term(out);
        for (;;)
        {
            char c = peek();
            if (c != '+' && c != '-')
                break;
            pos_++;
            mpz_t rhs;
            mpz_init(rhs);
            parse_term(rhs);
            if (c == '+')
                mpz_add(out, out, rhs);
            else
                mpz_sub(out, out, rhs);
            mpz_clear(rhs);
        }
    }

    // term := factor (('*'|'/'|'%') factor)*
    void parse_term(mpz_t out)
    {
        parse_factor(out);
        for (;;)
        {
            char c = peek();
            if (c != '*' && c != '/' && c != '%')
                break;
            pos_++;
            mpz_t rhs;
            mpz_init(rhs);
            parse_factor(rhs);
            if (c == '*')
            {
                mpz_mul(out, out, rhs);
            }
            else
            {
                if (mpz_sgn(rhs) == 0)
                {
                    mpz_clear(rhs);
                    fail("division by zero");
                }
                if (c == '/')
                    mpz_tdiv_q(out, out, rhs);
                else
                    mpz_tdiv_r(out, out, rhs);
            }
            mpz_clear(rhs);
        }
    }

    // factor := power
    void parse_factor(mpz_t out) { parse_power(out); }

    // power := unary ('^' power)?   (right-associative)
    void parse_power(mpz_t out)
    {
        parse_unary(out);
        if (peek() == '^')
        {
            pos_++;
            mpz_t exp;
            mpz_init(exp);
            parse_power(exp); // right-associative
            if (mpz_sgn(exp) < 0)
            {
                mpz_clear(exp);
                fail("negative exponent");
            }
            if (!mpz_fits_ulong_p(exp))
            {
                mpz_clear(exp);
                fail("exponent too large for mpz_pow_ui (must fit in unsigned long, i.e. < 2^32 on most platforms). "
                     "Note: 10^4294967295 would require ~4 GB just to store — are you sure you need an exponent that large?");
            }
            unsigned long e = mpz_get_ui(exp);
            mpz_clear(exp);
            mpz_pow_ui(out, out, e);
        }
    }

    // unary := ('+'|'-') unary | primary
    void parse_unary(mpz_t out)
    {
        char c = peek();
        if (c == '+')
        {
            pos_++;
            parse_unary(out);
        }
        else if (c == '-')
        {
            pos_++;
            parse_unary(out);
            mpz_neg(out, out);
        }
        else
        {
            parse_primary(out);
        }
    }

    // primary := NUMBER | '(' expr ')'
    void parse_primary(mpz_t out)
    {
        char c = peek();
        if (c == '(')
        {
            pos_++;
            parse_expr(out);
            if (peek() != ')')
                fail("expected ')'");
            pos_++;
        }
        else if (std::isdigit((unsigned char)c))
        {
            size_t start = pos_;
            while (pos_ < src_.size() && std::isdigit((unsigned char)src_[pos_]))
                pos_++;
            std::string num = src_.substr(start, pos_ - start);
            if (mpz_set_str(out, num.c_str(), 10) != 0)
                fail("invalid number literal");
        }
        else
        {
            fail("expected number or '('");
        }
    }
};
