import hymeko

def test_explicit_ir_loading():
    dsl_source = """
    Test_Graph {}
        fano {
            n0 {}
            n1 {}
            n2 {}

            @e0 { (~n0, ~n1); }
            @e1 { (~n0, ~n2); }
        }
    """

    # 1. Fázis: Biztonságos parszolás (Ha hiba van, itt megállunk, az Engine érintetlen marad)
    try:
        ir_topology = hymeko.PyHypergraphIR.from_dsl(dsl_source)
        print("[INFO] IR sikeresen generálva a LALRPOP által.")
    except SyntaxError as e:
        print(f"[FATAL] Szintaktikai hiba: {e}")
        return

    # 2. Fázis: Állapotgép inicializálása és IR explicit injektálása
    engine = hymeko.PyHypergraphEngine()
    engine.apply_ir(ir_topology)
    print("[INFO] IR topológia rátöltve a motorra.")

    # 3. Fázis: Memóriahíd és epoch zárás
    row_ptr, col_ind, val = engine.compile_epoch()
    print("[SUCCESS] CSR Mátrix zéró-másolattal elérhető!")

if __name__ == "__main__":
    test_explicit_ir_loading()