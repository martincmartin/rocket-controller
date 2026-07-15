import krpc

conn = krpc.connect(name="Synchronized Telemetry")

sc = conn.space_center
expr = conn.krpc.Expression
vessel = sc.active_vessel
refframe = vessel.orbit.body.reference_frame

# 1. Capture function calls as ProcedureCall objects
ut_call = conn.get_call(getattr, sc, "ut")
pos_call = conn.get_call(vessel.position, refframe)
vel_call = conn.get_call(vessel.velocity, refframe)
mass_call = conn.get_call(getattr, vessel, "mass")

# 2. Convert ProcedureCalls into Expressions
ut_expr = expr.call(ut_call)
pos_expr = expr.call(pos_call)
vel_expr = expr.call(vel_call)
mass_expr = expr.call(mass_call)

# 3. Combine expressions into a single Tuple Expression
tuple_expr = expr.create_tuple([ut_expr, pos_expr, vel_expr, mass_expr])

# 4. Pass the Expression DIRECTLY to add_stream (No second get_call wrapper!)
data_stream = conn.add_stream(tuple_expr)

# Reading loop
with data_stream.condition:
    data_stream.wait()

    while True:
        # Unpack the synchronized tuple evaluated on the server frame
        ut, position, velocity, mass = data_stream()

        print(f"UT: {ut:.2f}")
        print(f"Position: {position}")
        print(f"Velocity: {velocity}")
        print(f"Mass: {mass:.2f} kg")
        print("-" * 30)

        with data_stream.condition:
            data_stream.wait()
