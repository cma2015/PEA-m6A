# -*- coding: utf-8 -*-
# Copyright 2021 Minggui Song.
# All rights reserved.

from argparse import ArgumentParser, HelpFormatter
from importlib import import_module
from logging import getLogger
from os import getpid
from re import ASCII, compile
from sys import exit, stderr
from textwrap import wrap


from src.sys_output import Output

logger = getLogger(__name__)  # pylint: disable=invalid-name

class ScriptExecutor(object):
    """Loads the relevant script modules and executes the script.

    This class is initialised in each of the argparsers for the relevant
    command, then execute script is called within their set_default function.

    Attributes:
        - command (str): Full commands.
        - subparsers: Subparsers for each subcommand.
        - output: Output info, warning and error.

    """

    def __init__(self, command: str, subparsers=None) -> None:
        """Initialize ScriptExecutor.
        Args:
            - command (str): Full commands.
            - subparsers: Subparsers for each subcommand.

        """
        self.command = command.lower()
        self.subparsers = subparsers
        self.output = Output()

    def import_script(self):
        """Only import a script's modules when running that script."""
        # cmd = os.path.basename(sys.argv[0])
        src = 'src'
        mod = '.'.join((src, self.command.lower()))
        module = import_module(mod)
        script = getattr(module, self.command.title().replace('_', ''))
        return script

    def execute_script(self, arguments) -> None:
        """Run the script for called command."""
        self.output.info(f'Executing: {self.command}. PID: {getpid()}')
        logger.debug(f'Executing: {self.command}. PID: {getpid()}')

        try:
            script = self.import_script()
            process = script(arguments)
            process.process()
        except KeyboardInterrupt:  # pylint: disable=try-except-raise
            raise
        except SystemExit:
            pass
        except Exception:  # pylint: disable=broad-except
            logger.exception('Got Exception on main handler:')
            logger.critical(
                'An unexpected crash has occurred. '
                'Crash report written to logfile. '
                'Please verify you are running the latest version of *** '
                'before reporting.')
        finally:
            exit()

class FullHelpArgumentParser(ArgumentParser):
    """Identical to the built-in argument parser.

    On error it prints full help message instead of just usage information.
    """

    def error(self, message: str) -> None:
        """Print full help messages."""
        self.print_help(stderr)
        args = {'prog': self.prog, 'message': message}
        self.exit(2, f'{self.prog}: error: {message}\n')

class SmartFormatter(HelpFormatter):
    """Smart formatter for allowing raw formatting.

    Mainly acting in help text and lists in the helptext.

    To use: prefix the help item with 'R|' to overide
    default formatting. List items can be marked with 'L|'
    at the start of a newline.

    Adapted from: https://stackoverflow.com/questions/3853722
    """
    def __init__(self, prog: str,
                indent_increment: int = 2,
                max_help_position: int = 24,
                width=None) -> None:
        """Initialize SmartFormatter.

        Args:
            - prog (str): Program name.
            - indent_increment (int): Indent increment. default 2.
            - max_help_position (int): Max help position. default 24.
            - width: Width.

        """
        super().__init__(prog, indent_increment, max_help_position, width)
        self._whitespace_matcher_limited = compile(r'[ \r\f\v]+', ASCII)

    def _split_lines(self, text: str, width) -> list:
        if text.startswith('R|'):
            text = self._whitespace_matcher_limited.sub(' ', text).strip()[2:]
            output = []
            for txt in text.splitlines():
                indent = ''
                if txt.startswith('L|'):
                    indent = '    '
                    txt = '  - {}'.format(txt[2:])
                output.extend(wrap(
                    txt, width, subsequent_indent=indent))
            return output
        return HelpFormatter._split_lines(self, text, width)

class PEAM6AArgs(object):
    """PEA-m6A argument parser functions.

    It is universal to all commands.
    Should be the parent function of all subsequent argparsers.

    Attributes:
        - global_arguments: Global arguments.
        - argument_list: Argument list.
        - optional_arguments: Optional arguments.
        - parser: Parser.

    """

    def __init__(self, subparser, command: str,
                description: str = 'default', subparsers=None) -> None:
        """Initialize PEAM6AArgs.

        Args:
            - subparser: Subparser.
            - command (str): Command.
            - description (str): Description. default 'default'.
            - subparsers: Subparsers.

        """
        self.global_arguments = self.get_global_arguments()
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        if not subparser:
            return
        self.parser = self.create_parser(subparser, command, description)
        self.add_arguments()
        script = ScriptExecutor(command, subparsers)
        self.parser.set_defaults(func=script.execute_script)

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        return argument_list

    @staticmethod
    def get_optional_arguments() -> list:
        """Put the arguments in a list so that they are accessible.

        This is used for when there are sub-children.
        (e.g. convert and extract) Override this for custom arguments.
        """
        argument_list = []
        return argument_list

    @staticmethod
    def get_global_arguments() -> list:
        """Arguments that are used in ALL parts of PEA-m6A.

        DO NOT override this!
        """
        global_args = []
        global_args.append({'opts': ('-v', '--version'),
                            'action': 'version',
                            'version': 'PEA-m6A v0.0.1a'})
        return global_args

    @staticmethod
    def create_parser(subparser, command: str, description: str):
        """Create the parser for the selected command."""
        parser = subparser.add_parser(
            command,
            help=description,
            description=description,
            epilog='Questions and feedback: '
                ' https://github.com/Songmg-Nwafu/',
            formatter_class=SmartFormatter)
        return parser

    def add_arguments(self) -> None:
        """Parse the arguments passed in from argparse."""
        options = (self.global_arguments + self.argument_list +
                    self.optional_arguments)
        for option in options:
            args = option['opts']
            kwargs = {key: option[key]
                    for key in option.keys() if key != 'opts'}
            self.parser.add_argument(*args, **kwargs)

class DPArgs(PEAM6AArgs):
    """."""

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        argument_list.append({
            'opts': ('-i','--input'),
            'dest': 'input_dir',
            'default': '/home/malab9/Documents/project/01_m6A_prediction/01_data/12_PEA_Features/Contcat_features',
            'type': str,
            'help': 'Path to processed data directory.'})
        argument_list.append({
            'opts': ('-len','--instance_length'),
            'dest': 'instance_length',
            'default': 50,
            'type': int,
            'help': 'Instance length'})
        argument_list.append({
            'opts': ('-sl','--stride_length'),
            'dest': 'stride_length',
            'default': 20,
            'type': int,
            'help': 'Stride length'})
        argument_list.append({
            'opts': ('-o','--output_dir'),
            'dest': 'output_dir',
            'default': './Data',
            'type': str,
            'help': 'Path to output directory'})
        return argument_list

class SEArgs(PEAM6AArgs):
    """."""

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        argument_list.append({
            'opts': ('-s', '--species'),
            'dest': 'species',
            'required': True,
            'type': str,
            'help': 'input species name.'})
        argument_list.append({
            'opts': ('-ad', '--ann_dic'),
            'dest': 'ann_dic',
            'required': True,
            'type': str,
            'help': 'input annotation files folder.'})
        argument_list.append({
            'opts': ('-p', '--peak'),
            'dest': 'peak',
            'required': True,
            'type': str,
            'help': 'input peak bed file.'})
        argument_list.append({
            'opts': ('-o', '--output'),
            'dest': 'output',
            'required': True,
            'type': str,
            'help': 'output path.'})
        argument_list.append({
            'opts': ('-l', '--length'),
            'dest': 'length',
            'default': 500,
            'type': int,
            'help': 'positive and negative sample sequence length.'})
        argument_list.append({
            'opts': ('-m', '--motif'),
            'dest': 'motif',
            'default': 'RRACH',
            'type': str,
            'help': 'Motif.'})
        return argument_list

class FEArgs(PEAM6AArgs):
    """."""

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        argument_list.append({
            'opts': ('-e', '--encoding'),
            'dest': 'encoding',
            'choices':['onehot', 'statistics', 'deeplearning'],
            'required': True,
            'type': str,
            'help': 'input encoding strategy.'})
        argument_list.append({
            'opts': ('-s', '--species'),
            'dest': 'species',
            'required': True,
            'type': str,
            'help': 'input species name.'})
        argument_list.append({
            'opts': ('-i', '--input'),
            'dest': 'input',
            'required': True,
            'type': str,
            'help': 'input files path(i.e. .fa, .npy).'})
        argument_list.append({
            'opts': ('-o', '--output'),
            'dest': 'output',
            'required': True,
            'type': str,
            'help': 'output path.'})
        argument_list.append({
            'opts': ('-len','--instance_length'),
            'dest': 'instance_length',
            'default': 50,
            'type': int,
            'help': 'Instance length(required if using one-hot and deeplearning)'})
        argument_list.append({
            'opts': ('-sl','--stride_length'),
            'dest': 'stride_length',
            'default': 10,
            'type': int,
            'help': 'Stride length(required if using one-hot and deeplearning)'})
        argument_list.append({
            'opts': ('-mp','--model_path'),
            'dest': 'model_path',
            'type': str,
            'help': 'Deep learning model name(required if using deeplearning)'})
        return argument_list

class TrainArgs(PEAM6AArgs):
    """."""

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        argument_list.append({
            'opts': ('-i','--input'),
            'dest': 'input_dir',
            'default': '/home/malab9/Documents/project/03_m6ABoost/12_single_exon/02_zma_features_cdhit/03_weakrm',
            'type': str,
            'required': True,
            'help': 'Path to processed data directory.'})
        argument_list.append({
            'opts': ('-m','--model'),
            'dest': 'model_name',
            'default': 'WeakRM',
            'required': True,
            'choices':['WeakRM', 'PEAm6A'],
            'type': str,
            'help': 'One of [WeakRM, PEAm6A]'})
        argument_list.append({
            'opts': ('-matrix','--matrix_name'),
            'dest': 'matrix_name',
            'required': True,
            'default': 'DL',
            'choices':['DL', 'ST', 'OT'],
            'nargs': '+',
            'type': str,
            'help': 'Some of [DL, ST, OT]'})
        argument_list.append({
            'opts': ('-eval','--eval_after_train'),
            'dest': 'eval_after_train',
            'default': False,
            'type': bool,
            'help': 'Eval after train'})
        argument_list.append({
            'opts': ('-e','--epoch'),
            'dest': 'epoch',
            'default': 50,
            'type': int,
            'help': 'The number of epoch'})
        argument_list.append({
            'opts': ('-lr','--lr_init'),
            'dest': 'lr_init',
            'default': 1e-4,
            'type': float,
            'help': 'Initial learning rate'})
        argument_list.append({
            'opts': ('-ld','--lr_decay'),
            'dest': 'lr_decay',
            'default': 1e-5,
            'type': float,
            'help': 'Decayed learning rate'})
        argument_list.append({
            'opts': ('-len','--instance_length'),
            'dest': 'instance_length',
            'default': 50,
            'type': int,
            'help': 'Instance length'})
        argument_list.append({
            'opts': ('-sl','--stride_length'),
            'dest': 'stride_length',
            'default': 10,
            'type': int,
            'help': 'Stride length'})
        argument_list.append({
            'opts': ('-o','--cp_dir'),
            'dest': 'checkpoint_directory',
            'default': '/home/malab9/Documents/project/03_m6ABoost/12_single_exon/02_zma_features_cdhit/05_checkpoint',
            'type': str,
            'required': True,
            'help': 'Path to checkpoint directory'})
        argument_list.append({
            'opts': ('-cn','--cp_name'),
            'dest': 'checkpoint_name',
            'default': None,
            'required': True,
            'type': str,
            'help': 'Name of saved checkpoint'})
        return argument_list

class MAArgs(PEAM6AArgs):
    """."""

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        argument_list.append({
            'opts': ('-i', '--input'),
            'dest': 'input',
            'required': True,
            'type': str,
            'help': 'input dictory path.'})
        argument_list.append({
            'opts': ('-o', '--output'),
            'dest': 'output',
            'required': True,
            'type': str,
            'help': 'output dictory path.'})
        argument_list.append({
            'opts': ('-on', '--output_name'),
            'dest': 'output_name',
            'required': True,
            'type': int,
            'help': 'The name of output file'})
        argument_list.append({
            'opts': ('-model', '--model'),
            'dest': 'model',
            'required': True,
            'type': int,
            'help': 'models.'})
        argument_list.append({
            'opts': ('-plot','--plot'),
            'dest': 'plot',
            'required': True,
            'default': 'summary',
            'choices':['summary', 'dependence'],
            'type': str,
            'help': 'One of [summary, dependence]'})
        argument_list.append({
            'opts': ('-f1', '--features1'),
            'dest': 'features1',
            'type': str,
            'help': 'models.'})
        argument_list.append({
            'opts': ('-f2', '--features2'),
            'dest': 'features2',
            'type': str,
            'help': 'models.'})
        return argument_list

class PredictArgs(PEAM6AArgs):
    """."""

    @staticmethod
    def get_argument_list() -> list:
        """Put the arguments in a list so that they are accessible."""
        argument_list = []
        argument_list.append({
            'opts': ('-i','--input'),
            'dest': 'input_dir',
            'required': True,
            'type': str,
            'help': 'Path to processed data directory.'})
        argument_list.append({
            'opts': ('-matrix','--matrix_name'),
            'dest': 'matrix_name',
            'required': True,
            'default': 'DL',
            'choices':['DL', 'ST', 'OT'],
            'nargs': '+',
            'type': str,
            'help': 'Some of [DL, ST, OT]'})
        argument_list.append({
            'opts': ('-m','--model'),
            'dest': 'model_name',
            'required': True,
            'type': str,
            'help': 'input saved model name'})
        argument_list.append({
            'opts': ('-o','--ouput'),
            'dest': 'output',
            'required': True,
            'default': None,
            'type': str,
            'help': 'Name of saved checkpoint'})
        return argument_list