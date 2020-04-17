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

std::string LogicInduction::search() {
    std::vector<std::shared_ptr<LogicExpression>> exprs;
    for (size_t i = 0; i < m_config->nr_input_variables; ++i) {
        exprs.emplace_back(std::shared_ptr<LogicExpression>(new Literal(i, false)));
        exprs.emplace_back(std::shared_ptr<LogicExpression>(new Literal(i, true)));
    }
    if (_check_solution(exprs)) return m_solution->to_string(m_context);

    for (size_t depth = 1; depth < m_config->depth; ++depth) {
        std::vector<std::shared_ptr<LogicExpression>> new_exprs;
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

        if (_check_solution(new_exprs)) return m_solution->to_string(m_context);
        exprs.insert(exprs.end(), new_exprs.begin(), new_exprs.end());
    }

    return "";
}

bool LogicInduction::_check_solution(std::vector<std::shared_ptr<LogicExpression>> expr) {
    for (auto &&e: expr) {
        if (e->coverage(m_context) >= m_config->coverage) {
            m_solution = e;
            return true;
        }
    }
    return false;
}


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
