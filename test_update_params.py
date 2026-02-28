"""Tests for Model.update_params() method."""
import time
import pyomo.environ as pyo


def test_update_params_basic():
    """Test basic update_params functionality."""
    m = pyo.AbstractModel()
    m.N = pyo.Set()
    m.cost = pyo.Param(m.N, mutable=True, default=1.0)
    m.demand = pyo.Param(m.N, mutable=True, default=10.0)
    m.x = pyo.Var(m.N, domain=pyo.NonNegativeReals)

    def obj_rule(m):
        return sum(m.cost[n] * m.x[n] for n in m.N)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def demand_rule(m, n):
        return m.x[n] >= m.demand[n]
    m.demand_con = pyo.Constraint(m.N, rule=demand_rule)

    # Create initial instance
    data1 = {None: {
        'N': {None: [1, 2, 3]},
        'cost': {1: 1.0, 2: 2.0, 3: 3.0},
        'demand': {1: 10.0, 2: 20.0, 3: 30.0},
    }}
    instance = m.create_instance(data=data1)

    # Verify initial values
    assert pyo.value(instance.cost[1]) == 1.0
    assert pyo.value(instance.cost[2]) == 2.0
    assert pyo.value(instance.demand[1]) == 10.0
    assert pyo.value(instance.demand[3]) == 30.0

    # Update params
    data2 = {None: {
        'cost': {1: 5.0, 2: 6.0, 3: 7.0},
        'demand': {1: 100.0, 2: 200.0, 3: 300.0},
    }}
    result = instance.update_params(data=data2)

    # Verify updated values
    assert pyo.value(instance.cost[1]) == 5.0
    assert pyo.value(instance.cost[2]) == 6.0
    assert pyo.value(instance.cost[3]) == 7.0
    assert pyo.value(instance.demand[1]) == 100.0
    assert pyo.value(instance.demand[2]) == 200.0
    assert pyo.value(instance.demand[3]) == 300.0

    # Verify method chaining
    assert result is instance

    print("PASS: test_update_params_basic")


def test_update_params_ignores_non_params():
    """Test that update_params silently ignores non-Param data."""
    m = pyo.AbstractModel()
    m.N = pyo.Set()
    m.cost = pyo.Param(m.N, mutable=True, default=1.0)
    m.x = pyo.Var(m.N, domain=pyo.NonNegativeReals)

    data = {None: {
        'N': {None: [1, 2, 3]},
        'cost': {1: 1.0, 2: 2.0, 3: 3.0},
    }}
    instance = m.create_instance(data=data)

    # Include Set data in update - should be silently ignored
    update_data = {None: {
        'N': {None: [4, 5, 6]},  # Set data - should be ignored
        'cost': {1: 10.0, 2: 20.0, 3: 30.0},
    }}
    instance.update_params(data=update_data)

    assert pyo.value(instance.cost[1]) == 10.0
    # Set should NOT have changed
    assert list(instance.N) == [1, 2, 3]

    print("PASS: test_update_params_ignores_non_params")


def test_update_params_immutable_error():
    """Test that update_params raises error for immutable params."""
    m = pyo.AbstractModel()
    m.N = pyo.Set()
    m.cost = pyo.Param(m.N, default=1.0)  # NOT mutable

    data = {None: {
        'N': {None: [1, 2, 3]},
        'cost': {1: 1.0, 2: 2.0, 3: 3.0},
    }}
    instance = m.create_instance(data=data)

    update_data = {None: {
        'cost': {1: 10.0},
    }}
    try:
        instance.update_params(data=update_data)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "immutable" in str(e).lower()

    print("PASS: test_update_params_immutable_error")


def test_update_params_unconstructed_error():
    """Test that update_params raises error on unconstructed model."""
    m = pyo.AbstractModel()
    m.N = pyo.Set()
    m.cost = pyo.Param(m.N, mutable=True, default=1.0)

    try:
        m.update_params(data={None: {'cost': {1: 10.0}}})
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "unconstructed" in str(e).lower()

    print("PASS: test_update_params_unconstructed_error")


def test_update_params_constraint_expressions_reflect_changes():
    """Test that constraint expressions reflect updated param values."""
    m = pyo.AbstractModel()
    m.N = pyo.Set()
    m.demand = pyo.Param(m.N, mutable=True, default=10.0)
    m.x = pyo.Var(m.N, domain=pyo.NonNegativeReals)

    def demand_rule(m, n):
        return m.x[n] >= m.demand[n]
    m.demand_con = pyo.Constraint(m.N, rule=demand_rule)

    data1 = {None: {
        'N': {None: [1, 2]},
        'demand': {1: 10.0, 2: 20.0},
    }}
    instance = m.create_instance(data=data1)

    # Check initial constraint lower bound
    assert pyo.value(instance.demand_con[1].lower) == 10.0
    assert pyo.value(instance.demand_con[2].lower) == 20.0

    # Update demand
    data2 = {None: {'demand': {1: 100.0, 2: 200.0}}}
    instance.update_params(data=data2)

    # Constraint expressions should reflect the new values
    assert pyo.value(instance.demand_con[1].lower) == 100.0
    assert pyo.value(instance.demand_con[2].lower) == 200.0

    print("PASS: test_update_params_constraint_expressions_reflect_changes")


def test_update_params_partial_update():
    """Test that update_params can update only some params."""
    m = pyo.AbstractModel()
    m.N = pyo.Set()
    m.cost = pyo.Param(m.N, mutable=True, default=1.0)
    m.demand = pyo.Param(m.N, mutable=True, default=10.0)

    data = {None: {
        'N': {None: [1, 2]},
        'cost': {1: 1.0, 2: 2.0},
        'demand': {1: 10.0, 2: 20.0},
    }}
    instance = m.create_instance(data=data)

    # Only update demand, not cost
    update_data = {None: {'demand': {1: 100.0, 2: 200.0}}}
    instance.update_params(data=update_data)

    # Cost should be unchanged
    assert pyo.value(instance.cost[1]) == 1.0
    assert pyo.value(instance.cost[2]) == 2.0
    # Demand should be updated
    assert pyo.value(instance.demand[1]) == 100.0
    assert pyo.value(instance.demand[2]) == 200.0

    print("PASS: test_update_params_partial_update")


def test_update_params_performance():
    """Benchmark update_params vs create_instance."""
    N_SIZE = 100
    T_SIZE = 24
    ITERS = 20

    N_set = list(range(N_SIZE))
    T_set = list(range(T_SIZE))
    cost_data = {n: 1.0 + n * 0.1 for n in N_set}
    capacity_data = {n: 100.0 for n in N_set}

    m = pyo.AbstractModel()
    m.N = pyo.Set()
    m.T = pyo.Set()
    m.cost = pyo.Param(m.N, default=1.0)
    m.demand = pyo.Param(m.T, mutable=True, default=10.0)
    m.capacity = pyo.Param(m.N, default=100.0)
    m.x = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)
    m.y = pyo.Var(m.N, domain=pyo.Binary)

    def obj_rule(m):
        return sum(m.cost[n] * m.x[n, t] for n in m.N for t in m.T)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def demand_rule(m, t):
        return sum(m.x[n, t] for n in m.N) >= m.demand[t]
    m.demand_con = pyo.Constraint(m.T, rule=demand_rule)

    def capacity_rule(m, n, t):
        return m.x[n, t] <= m.capacity[n] * m.y[n]
    m.capacity_con = pyo.Constraint(m.N, m.T, rule=capacity_rule)

    # Benchmark create_instance
    t0 = time.perf_counter()
    for day in range(ITERS):
        demand_data = {t: 10.0 + day for t in T_set}
        data = {None: {
            'N': {None: N_set}, 'T': {None: T_set},
            'cost': cost_data, 'demand': demand_data, 'capacity': capacity_data,
        }}
        inst = m.create_instance(data=data)
    t_create = time.perf_counter() - t0

    # Benchmark update_params
    demand_data = {t: 10.0 for t in T_set}
    data = {None: {
        'N': {None: N_set}, 'T': {None: T_set},
        'cost': cost_data, 'demand': demand_data, 'capacity': capacity_data,
    }}
    instance = m.create_instance(data=data)

    t0 = time.perf_counter()
    for day in range(ITERS):
        demand_data = {t: 10.0 + day for t in T_set}
        update_data = {None: {'demand': demand_data}}
        instance.update_params(data=update_data)
    t_update = time.perf_counter() - t0

    speedup = t_create / t_update
    print(f"\nPerformance (N={N_SIZE}, T={T_SIZE}, {ITERS} iterations):")
    print(f"  create_instance:  {t_create:.4f}s ({t_create/ITERS*1000:.1f}ms/iter)")
    print(f"  update_params:    {t_update:.4f}s ({t_update/ITERS*1000:.1f}ms/iter)")
    print(f"  Speedup:          {speedup:.0f}x")

    assert speedup > 10, f"Expected >10x speedup, got {speedup:.1f}x"
    print("PASS: test_update_params_performance")


def test_update_params_on_concrete_model():
    """Test that update_params works on ConcreteModel too."""
    m = pyo.ConcreteModel()
    m.N = pyo.Set(initialize=[1, 2, 3])
    m.cost = pyo.Param(m.N, initialize={1: 1.0, 2: 2.0, 3: 3.0}, mutable=True)

    m.update_params(data={None: {'cost': {1: 10.0, 2: 20.0, 3: 30.0}}})

    assert pyo.value(m.cost[1]) == 10.0
    assert pyo.value(m.cost[2]) == 20.0
    assert pyo.value(m.cost[3]) == 30.0

    print("PASS: test_update_params_on_concrete_model")


if __name__ == '__main__':
    test_update_params_basic()
    test_update_params_ignores_non_params()
    test_update_params_immutable_error()
    test_update_params_unconstructed_error()
    test_update_params_constraint_expressions_reflect_changes()
    test_update_params_partial_update()
    test_update_params_performance()
    test_update_params_on_concrete_model()
    print("\nAll tests passed!")
