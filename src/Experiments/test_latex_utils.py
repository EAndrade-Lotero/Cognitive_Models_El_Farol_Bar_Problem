from Utils.LaTeX_utils import PrintLaTeX


def test_print_parameters():
    parameters = {
        "threshold": 0.5,
        "num_agents": 2,
        "inverse_temperature": 50,
        "go_drive": 0.5,
        "wsls_strength": 1
    }

    cadena = PrintLaTeX.print_parameters(parameters)
    print(cadena)