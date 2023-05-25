import argparse

if __name__ == "__main__":
    # create the top-level parser
    my_parser = argparse.ArgumentParser(
        prog="PROG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # create sub-parser
    sub_parsers = my_parser.add_subparsers(
        title="Operating modes",
        description="Select the operating mode",
        dest="mode",
        required=True,
    )

    # create the parser for the "agent" sub-command
    parser_agent = sub_parsers.add_parser("agent", help="Agent mode")
    parser_agent.add_argument(
        "--db_server", type=str, help="DB server name", default="localhost"
    )
    parser_agent.add_argument(
        "--update_interval",
        type=int,
        help="Interval of updating policy parameters",
        default=64,
    )

    # create the parse for the "learner" sub-command
    parser_learner = sub_parsers.add_parser("learner", help="Learner mode")
    parser_learner.add_argument(
        "-e",
        "--environment",
        type=str,
        help="Only OpenAI Gym/PyBullet environments are available!",
        required=True,
    )
    parser_learner.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Number of agent's steps",
        default=int(1e6),
    )
    
    # create the parse for the "tester" sub-command
    parser_tester = sub_parsers.add_parser("tester", help="Tester mode")
    parser_tester.add_argument(
        "-t",
        "--max_steps",
        type=int,
        help="Number of agent's steps",
        default=int(1e6),
    )
    parser_tester.add_argument(
        "--render", action="store_true", help="Render the environment"
    )
    parser_tester.add_argument(
        "-f", "--model_path", type=str, help="Path to saved model"
    )

    args = my_parser.parse_args()

    print(args)

    if args.mode == "agent":
        print('mode 1')  
    elif args.mode == "learner":  
        print('mode 2')  
    elif args.mode == "tester":
        print('mode 3')    