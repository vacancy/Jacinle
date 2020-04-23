/*
 * logic_induction.h
 * Copyright (C) 2020 Jiayuan Mao <maojiayuan@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef LOGIC_INDUCTION_H
#define LOGIC_INDUCTION_H

#include <vector>
#include <string>
#include <sstream>
#include <memory>

enum LogicFormType {
    CONJUNCTION_TYPE = 1,
    DISJUNCTION_TYPE = 2,
    GENERAL_TYPE = 3
};

enum LogicFormOutputFormat {
    DEFAULT_FORMAT = 1,
    LISP_FORMAT = 2
};

class LogicInductionConfig {
public:
    LogicFormType type;
    LogicFormOutputFormat output_format;
    size_t nr_examples;
    size_t nr_input_variables;
    size_t nr_output_variables;
    size_t depth;
    double coverage;
};

class LogicInductionContext {
public:
    LogicInductionConfig *config;
    unsigned char *inputs;   // of size nr_examples * nr_input_variables.
    unsigned char *outputs;  // of size nr_examples * nr_output_variables.
    std::vector<std::string> input_names;
};

class LogicExpression {
public:
    LogicExpression() : m_result(), m_evaluated(false), m_coverage(-1) {};
    virtual ~LogicExpression() = default;

    const std::vector<bool> &eval(LogicInductionContext *ctx);
    double coverage(LogicInductionContext *ctx);
    virtual void eval_impl(LogicInductionContext *ctx) = 0;
    virtual std::string to_string(LogicInductionContext *ctx) const = 0;

    virtual bool is_literal() const { return false; }
    virtual const std::string type_str() const { return "Expr"; }
protected:
    std::vector<bool> m_result;
    bool m_evaluated;
    double m_coverage;
};

class Literal : public LogicExpression {
public:
    Literal(size_t index, bool negate) : m_index(index), m_negate(negate) {}

    virtual void eval_impl(LogicInductionContext *ctx) override;
    virtual std::string to_string(LogicInductionContext *ctx) const override;

    virtual const std::string type_str() const override { return "LITERAL"; }
    virtual bool is_literal() const override { return true; }

protected:
    size_t m_index;
    bool m_negate;
};

class UnaryLogicOp : public LogicExpression {
public:
    UnaryLogicOp(std::shared_ptr<LogicExpression> oprand) : m_oprand(oprand) {}
protected:
    std::shared_ptr<LogicExpression> m_oprand;
};

class Not : public UnaryLogicOp {
public:
    using UnaryLogicOp::UnaryLogicOp;

    virtual void eval_impl(LogicInductionContext *ctx) override;
    virtual std::string to_string(LogicInductionContext *ctx) const override;

    virtual const std::string type_str() const override { return "NOT"; }
};

class BinaryLogicOp : public LogicExpression {
public:
    BinaryLogicOp(std::shared_ptr<LogicExpression> lhs, std::shared_ptr<LogicExpression> rhs) : m_lhs(lhs), m_rhs(rhs) {}
protected:
    std::shared_ptr<LogicExpression> m_lhs, m_rhs;
};

class And : public BinaryLogicOp {
public:
    using BinaryLogicOp::BinaryLogicOp;

    virtual void eval_impl(LogicInductionContext *ctx) override;
    virtual std::string to_string(LogicInductionContext *ctx) const override;

    virtual const std::string type_str() const override { return "AND"; }
};

class Or: public BinaryLogicOp {
public:
    using BinaryLogicOp::BinaryLogicOp;

    virtual void eval_impl(LogicInductionContext *ctx) override;
    virtual std::string to_string(LogicInductionContext *ctx) const override;

    virtual const std::string type_str() const override { return "AND"; }
};

class LogicInduction {
public:
    LogicInduction(LogicInductionConfig *config, LogicInductionContext *ctx) : m_config(config), m_context(ctx), m_solution() {}
    std::string search();

private:
    LogicInductionConfig *m_config;
    LogicInductionContext *m_context;
    std::shared_ptr<LogicExpression> m_solution;
};

#endif /* !LOGIC_INDUCTION_H */
