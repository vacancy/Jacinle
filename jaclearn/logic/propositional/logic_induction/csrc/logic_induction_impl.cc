/*
 * File   : logic_induction.cc
 * Author : Jiayuan Mao
 * Email  : maojiayuan@gmail.com
 * Date   : 04/16/2020
 *
 * Distributed under terms of the MIT license.
 */

#include "logic_induction.h"
#include <iostream>
#include <array>

#define FAST_SEARCH

const std::vector<bool> &LogicExpression::eval(LogicInductionContext *ctx) {
    if (!m_evaluated) {
        m_result.resize(ctx->config->nr_examples);
        m_evaluated = true;
        eval_impl(ctx);
    }
    return m_result;
}

double LogicExpression::coverage(LogicInductionContext *ctx) {
    if (m_coverage < 0) {
        size_t ans = 0;

        const auto &result = eval(ctx);
        for (size_t i = 0; i < ctx->config->nr_examples; ++i) {
            ans += (ctx->outputs[i] == result[i]);
        }

        m_coverage = double(ans) / ctx->config->nr_examples;
    }
    return m_coverage;
}

void Literal::eval_impl(LogicInductionContext *ctx) {
    for (size_t i = 0; i < ctx->config->nr_examples; ++i) {
        m_result[i] = ctx->inputs[ctx->config->nr_input_variables * i + m_index] ^ m_negate;
    }
}

std::string Literal::to_string(LogicInductionContext *ctx) const {
    if (m_negate) return "NOT " + ctx->input_names[m_index];
    else return ctx->input_names[m_index];
}

void Not::eval_impl(LogicInductionContext *ctx) {
    const auto &input = m_oprand->eval(ctx);
    for (size_t i = 0; i < ctx->config->nr_examples; ++i) {
        m_result[i] = !input[i];
    }
}

std::string Not::to_string(LogicInductionContext *ctx) const {
    if (ctx->config->output_format == LISP_FORMAT)
        return "NOT " + m_oprand->to_string(ctx);
    else
        return "(NOT " + m_oprand->to_string(ctx) + ")";
}

void And::eval_impl(LogicInductionContext *ctx) {
    const auto &input1 = m_lhs->eval(ctx);
    const auto &input2 = m_rhs->eval(ctx);
    for (size_t i = 0; i < ctx->config->nr_examples; ++i) {
        m_result[i] = input1[i] && input2[i];
    }
}

std::string And::to_string(LogicInductionContext *ctx) const {
    if (ctx->config->output_format == LISP_FORMAT)
        return "AND " + m_lhs->to_string(ctx) + " " + m_rhs->to_string(ctx) + ")";
    else
        return "(" + m_lhs->to_string(ctx) + " AND " + m_rhs->to_string(ctx) + ")";
}

void Or::eval_impl(LogicInductionContext *ctx) {
    const auto &input1 = m_lhs->eval(ctx);
    const auto &input2 = m_rhs->eval(ctx);
    for (size_t i = 0; i < ctx->config->nr_examples; ++i) {
        m_result[i] = input1[i] || input2[i];
    }
}

std::string Or::to_string(LogicInductionContext *ctx) const {
    if (ctx->config->output_format == LISP_FORMAT)
        return "OR " + m_lhs->to_string(ctx) + " " + m_rhs->to_string(ctx) + ")";
    else
        return "(" + m_lhs->to_string(ctx) + " OR " + m_rhs->to_string(ctx) + ")";

}

#ifndef FAST_SEARCH

static std::shared_ptr<LogicExpression> _check_solution(const std::vector<std::shared_ptr<LogicExpression>> &expr, LogicInductionContext *m_context, LogicInductionConfig *config) {
    for (auto &&e: expr) {
        if (e->coverage(m_context) >= m_config->coverage) {
            return e;
        }
    }
    return nullptr;
}

std::string LogicInduction::search() {
    using expr_vec = std::vector<std::shared_ptr<LogicExpression>>;
    expr_vec exprs;
    for (size_t i = 0; i < m_config->nr_input_variables; ++i) {
        exprs.emplace_back(std::shared_ptr<LogicExpression>(new Literal(i, false)));
        exprs.emplace_back(std::shared_ptr<LogicExpression>(new Literal(i, true)));
    }
    auto solution = _check_solution(exprs, m_context, m_config);
    if (solution) return solution->to_string(m_context);

    for (size_t depth = 1; depth < m_config->depth; ++depth) {
        expr_vec new_exprs;
        if (m_config->type == GENERAL_TYPE) {
            for (auto &&e: exprs) {
                if (e->type_str() != "NOT" and e->type_str() != "LITERAL") {
                    new_exprs.emplace_back(std::shared_ptr<LogicExpression>(new Not(e)));
                }
            }
        }
        for (auto &&e1: exprs) for (auto &&e2: exprs) {
            if (m_config->type == GENERAL_TYPE || m_config->type == CONJUNCTION_TYPE) {
                if (e1->type_str() != "AND") {
                    new_exprs.emplace_back(std::shared_ptr<LogicExpression>(new And(e1, e2)));
                }
            }
            if (m_config->type == GENERAL_TYPE || m_config->type == DISJUNCTION_TYPE) {
                if (e1->type_str() != "OR") {
                    new_exprs.emplace_back(std::shared_ptr<LogicExpression>(new Or(e1, e2)));
                }
            }
        }

        auto solution = _check_solution(new_exprs, m_context, m_config);
        if (solution) return solution->to_string(m_context);
        exprs.insert(exprs.end(), new_exprs.begin(), new_exprs.end());

        std::cerr << "depth=" << depth + 1 << " #exprs=" << exprs.size()<< std::endl;
    }

    return "";
}

#else

using expr_vec = std::vector<std::shared_ptr<LogicExpression>>;
using expr_arrvec = std::array<expr_vec, 4>;

static std::shared_ptr<LogicExpression> _check_solution(const expr_arrvec &expr, LogicInductionContext *m_context, LogicInductionConfig *m_config) {
    for (int i: {0, 1, 2, 3}) for (auto &&e: expr[i]) {
        if (e->coverage(m_context) >= m_config->coverage) {
            return e;
        }
    }
    return nullptr;
}

std::string LogicInduction::search() {
    expr_arrvec exprs;
    for (size_t i = 0; i < m_config->nr_input_variables; ++i) {
        exprs[0].emplace_back(std::shared_ptr<LogicExpression>(new Literal(i, false)));
        exprs[0].emplace_back(std::shared_ptr<LogicExpression>(new Literal(i, true)));
    }

    auto solution = _check_solution(exprs, m_context, m_config);
    if (solution) return solution->to_string(m_context);

    for (size_t depth = 1; depth < m_config->depth; ++depth) {
        expr_arrvec new_exprs;
        if (m_config->type == GENERAL_TYPE) {
            std::cerr << "depth=" << depth + 1 << " NOT #exprs=" << exprs[2].size() << std::endl;
            for (auto &&e: exprs[2]) {
                new_exprs[1].emplace_back(std::shared_ptr<LogicExpression>(new Not(e)));
            }
            std::cerr << "depth=" << depth + 1 << " NOT #exprs=" << exprs[3].size() << std::endl;
            for (auto &&e: exprs[3]) {
                new_exprs[1].emplace_back(std::shared_ptr<LogicExpression>(new Not(e)));
            }
        }
        if (m_config->type == GENERAL_TYPE || m_config->type == CONJUNCTION_TYPE) {
            for (int e1t: {0, 1, 3}) for (int e2t: {0, 1, 2, 3}) {
                std::cerr << "depth=" << depth + 1 << " AND #exprs=" << exprs[e1t].size() << ", " << exprs[e2t].size() << std::endl;
                for (auto &&e1: exprs[e1t]) for (auto &&e2: exprs[e2t]) {
                    auto e = std::shared_ptr<LogicExpression>(new And(e1, e2));
                    if (depth < m_config->depth - 1) {
                        new_exprs[2].emplace_back(e);
                    } else {
                        if (e->coverage(m_context) >= m_config -> coverage) {
                            return e->to_string(m_context);
                        }
                    }
                }
            }
        }
        if (m_config->type == GENERAL_TYPE || m_config->type == DISJUNCTION_TYPE) {
            for (int e1t: {0, 1, 2}) for (int e2t: {0, 1, 2, 3}) {
                std::cerr << "depth=" << depth + 1 << " OR #exprs=" << exprs[e1t].size() << ", " << exprs[e2t].size() << std::endl;
                for (auto &&e1: exprs[e1t]) for (auto &&e2: exprs[e2t]) {
                    auto e = std::shared_ptr<LogicExpression>(new Or(e1, e2));
                    if (depth < m_config->depth - 1) {
                        new_exprs[3].emplace_back(e);
                    } else {
                        if (e->coverage(m_context) >= m_config -> coverage) {
                            return e->to_string(m_context);
                        }
                    }
                }
            }
        }

        auto solution = _check_solution(new_exprs, m_context, m_config);
        if (solution) return solution->to_string(m_context);

        for (int i: {0, 1, 2, 3}) {
            exprs[i].insert(exprs[i].end(), new_exprs[i].begin(), new_exprs[i].end());
        }
    }

    return "";
}

#endif


#if false
int main() {
    LogicInductionConfig config {
        LogicFormType::GENERAL,
        LogicFormOutputFormat::DEFAULT,
        /* nr_examples = */ 4,
        /* nr_input_variables = */ 4,
        /* nr_output_variables = */ 1,
        /* depth = */ 4,
        /* coverage = */ 0.99
    };
    LogicInductionContext context {
        &config,
        std::vector<bool> { 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1 },
        std::vector<bool> { 0, 1, 1, 0 },
        std::vector<std::string> { "x", "y" }
    };
    auto induction = LogicInduction(&config, &context);
    std::cerr << induction.search() << std::endl;

    return 0;
}
#endif
