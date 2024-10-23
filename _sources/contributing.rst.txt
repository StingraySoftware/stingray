===================================
Get Help, Report Bugs or Contribute
===================================

Reporting Bugs and Issues, Getting Help, Providing Feedback
===========================================================

We would love to hear from you!
We are writing Stingray to be useful to you, so if you encounter problems, have questions, would like to request features or just want to chat with us, please don't hesitate to get in touch!

The best and easiest way to get in touch with us regarding bugs and issues is the GitHub `Issues page <https://github.com/StingraySoftware/stingray/issues>`_.
If you're not sure whether what you've encountered is a bug, if you have any questions or need advice getting some of the code to run, or would like to request a feature or suggest additions/changes, you can also contact us via the Slack group or our mailing list.

Please use `this link <https://join.slack.com/t/stingraysoftware/shared_invite/zt-49kv4kba-mD1Y~s~rlrOOmvqM7mZugQ>`_ to join Slack or send `one of us <https://github.com/orgs/StingraySoftware/people>`_ an email to join the mailing list.

Getting Involved with Development
=================================

We encourage you to get involved with Stingray in any way you can!
First, read through the `README <https://github.com/StingraySoftware/stingray/blob/master/README.rst>`_.
Then, fork the `stingray <https://github.com/StingraySoftware/stingray>`_ and `notebooks <https://github.com/StingraySoftware/notebooks>`_ repositories (if you need a primer on GitHub and git version control, `look here <https://www.webpagefx.com/blog/web-design/git-tutorials-beginners/>`_) and work your way through the Jupyter notebook tutorials for the main modules.
Once you've familiarized yourself with the basics of Stingray, go to the `Stingray issues page <https://github.com/StingraySoftware/stingray>`_ and try to tackle one!
Finally, you can read `these slides <https://speakerdeck.com/abigailstev/time-series-exploration-with-stingray>`_ from a talk on Stingray in 2021 at the 9th Microquasar Workshop.

For organizing and coordinating the software development, we have a Slack group and a mailing list -- please use `this link <https://join.slack.com/t/stingraysoftware/shared_invite/zt-49kv4kba-mD1Y~s~rlrOOmvqM7mZugQ>`_ for Slack or send `one of us <https://github.com/orgs/StingraySoftware/people>`_ an email to join.


Contributing to Stingray
========================

   All great things have small beginnings.

Hello there! We love and appreciate every small contribution you can
make to improve Stingray! We are proudly open source and believe
our(yes! yours as well) work will help enhance the quality of research
around the world. We want to make contributing to stingray as easy and
transparent as possible, whether it’s:

-  Reporting a bug
-  Discussing the current state of the code
-  Submitting a fix
-  Proposing new features

A successful project is not just built by amazing programmers but by the
combined, unrelenting efforts of coders, testers, reviewers, and
documentation writers. There are a few guidelines that we need all
contributors to follow so that we can have a chance of keeping on top of
things.

Contribution Guidelines
-----------------------

Contributions from everyone, experienced and inexperienced, are welcome!
If you don’t know where to start, look at the `Open
Issues <https://github.com/StingraySoftware/stingray/issues>`__ and/or
get involved in our `Slack
channel <https://join.slack.com/t/stingraysoftware/shared_invite/zt-49kv4kba-mD1Y~s~rlrOOmvqM7mZugQ>`__. This code is
written in Python 3.8+, but in general we will follow the Astropy/ Numpy
minimum Python versions. Tests run at each commit during Pull Requests,
so it is easy to single out points in the code that break this
compatibility.

-  **Branches:**

   -  Don’t use your main **branch (forked) for anything. Consider
      deleting your main** branch.
   -  Make a new branch, called a feature branch, for each separable set
      of changes: “one task, one branch”.
   -  Start that new feature branch from the most current development
      version of stingray.
   -  Name of branch should be the purpose of change eg.
      *bugfix-for-issue20* or *refactor-lightcurve-code.*
   -  Never merge changes from stingray/main into your feature branch.
      If changes in the development version require changes to our code
      you can rebase, but only if asked.

-  **Commits:**

   -  Make frequent commits.
   -  One commit per logical change in the code-base.
   -  Add commit message.

-  **Naming Conventions:**

   -  Change name of the remote origin(*yourusername/stingray*) to your
      *github-username.*
   -  Name the remote that is the primary stingray repository(
      *StingraySoftware/stingray*) as stingray.

Contribution Workflow
~~~~~~~~~~~~~~~~~~~~~

These, conceptually, are the steps you will follow in contributing to
Stingray. These steps keep work well organized, with readable history.
This in turn makes it easier for project maintainers (that might be you)
to see what you’ve done, and why you did it:

1. Regularly fetch latest stingray development version ``stingray/main``
   from GitHub.
2. Make a new feature branch. **Recommended:** Use virtual environments
   to work on branch.
3. Editing Workflow:

   1. One commit per logical change.
   2. Run tests to make sure that changes don’t break existing code.
   3. Code should have appropriate docstring.
   4. Format code appropriately, use ``black`` as described below.
   5. Update appropriate documentation if necessary and test it on
      sphinx.
   6. Write tests that cover all code changes.
   7. If modifications require more than one commit, break changes into
      smaller commits. Commits involving just the docs might use ``[docs only]`` in
      their commit message to avoid running all the tests. *Very* trivial commits
      (e.g. a space in a docstring) might skip *all* tests with ``[skip ci]`` in
      their commit message.
   8. Write a changelog entry in ``towncrier`` format (see below)
   9. Push the code on your remote(forked) repository.

4. All code changes should be submitted via PRs (i.e. fork, branch, work
   on stuff, just submit pull request). Code Reviews are super-useful:
   another contributor can review the code, which means both the
   contributor and reviewer will be up to date with how everything fits
   together, and can get better by reading each other’s code! :)
5. Take feedback and make changes/revise the PR as asked.

Coding Guidelines
-----------------

Compatibility and Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Compatibility:** All code must be compatible with **Python 3.8**
   **or later**, and with the **latest two major releases of Astropy**.
-  **Dependency Management:**

   -  The core package and affiliated packages should be importable with
      no dependencies other than the `Python Standard
      Library <https://docs.python.org/3/library/index.html>`__,
      `astropy <https://docs.astropy.org/en/stable/>`__>=4.0,
      `numpy <https://numpy.org/doc/stable/>`__>=1.17.0,
      `scipy <https://docs.scipy.org/doc/scipy/>`__>=1.1,
      `matplotlib <https://matplotlib.org/contents.html>`__>=3.0
   -  Additional dependencies are allowed for sub-modules or in function
      calls, but they must be noted in the package documentation and
      should only affect the relevant component. In functions and
      methods, the optional dependency should use a normal ``import``
      statement, which will raise an ``ImportError`` if the dependency
      is not available.

Coding Style and Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Style Guide:**

   -  Follow the `PEP8 style
      guide <https://www.python.org/dev/peps/pep-0008/>`__. Follow the
      existing coding style within the sub-package and avoid changes
      that are purely stylistic.
   -  Indentation should be **ONLY** with **four spaces** no mixing of
      tabs-and-spaces.
   -  Maximum line length should be **100** characters unless doing so
      makes the code unreadable, ugly.
   -  Functions and methods should be lower-case only, and separated by
      a ``_`` in case of multiple words eg. ``my_new_method``.
   -  Use verbose variable names (readability > economy). Only loop
      iteration variables are allowed to be a single letter.
   -  Classes start with an upper-case letter and use CamelCase eg.
      ``MyNewClass``.
   -  Inline comments should start with two spaces and a single #.

- **Formatting Style:** The new Python 3 formatting style should be
  used, i.e., f-strings ``f"{variable_name}"`` or
  ``"{0}".format(variable_name}`` should be used instead of
  ``"%s" % (variable_name)``. Additionally, the project enforces
  code formatting and style checks through the **pre-commit** tool,
  ensuring consistency and adherence to style guidelines across contributions.

- To set up pre-commit locally for the Stingray project, follow these steps:

  1. Install the pre-commit package:

     .. code-block:: bash

        $ pip install pre-commit

  2. Run pre-commit on all files in the Stingray repository:

     .. code-block:: bash

        $ pre-commit run --all-files

     This will run the pre-commit tools on all files in the Stingray git repository. The tools may automatically modify some files, while in other cases, they will report issues that require manual correction. If pre-commit makes changes to any files, those changes will appear as new modifications, which need to be staged before committing.


   

-  **Linter/Style Guide Checker:** Our testing infrastructure currently
   enforces a subset of the PEP8 style guide. You can check locally
   whether your changes have followed these by running
   `flake8 <https://pypi.org/project/flake8/>`__ with the following
   command:

   ``flake8 astropy --count --select=E101,W191,W291,W292,W293,W391,E111,E112,E113,E30,E502,E722,E901,E902,E999,F822,F823``

-  **Code Formatters:** We follow Astropy, enforcing this style guide
   using the black code formatter, see `The Black Code
   Style <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`__
   for details. Please run

   ``black stingray``

   before each commit

-  **Imports:**

   -  Absolute imports are to be used in general. The exception to this
      is relative imports of the form ``from . import modulename``, this
      convention makes it clearer what code is from the current
      sub-module as opposed to from another. It is best to use when
      referring to files within the same sub-module.
   -  The import ``numpy as np``, ``import scipy as sp``,
      ``import matplotlib as mpl``, and
      ``import matplotlib.pyplot as plt`` naming conventions should be
      used wherever relevant. ``from packagename import *`` should never
      be used, except as a tool to flatten the namespace of a module.

-  **Variable access in Classes:**

   -  Classes should either use direct variable access, or Python’s
      property mechanism for setting object instance variables.
      ``get_value/set_value`` style methods should be used only when
      getting and setting the values requires a
      computationally-expensive operation.
   -  Attribute names should be descriptive if possible, use names of
      desserts otherwise (e.g. for dummy test classes)

-  **super() function:** Classes should use the built-in ``super()``
   function when making calls to methods in their super-class(es) unless
   there are specific reasons not to. ``super()`` should be used
   consistently in all sub-classes since it does not work otherwise.

-  **Multiple Inheritance:** Multiple inheritance should be avoided in
   general without good reason.

-  **init.py:** The ``__init__.py`` files for modules should not contain
   any significant implementation code. ``__init__.py`` can contain
   docstrings and code for organizing the module layout, however if a
   module is small enough that it fits in one file, it should simply be
   a single file, rather than a directory with an ``__init__.py`` file.

Standard output, warnings, and errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Print Statement:** Used only for outputs in methods and scenarios
   explicitly requested by the user
-  **Errors and Exceptions:** Always use the ``raise`` with built-in or
   custom exception classes. The nondescript ``Exception`` class should
   be avoided as much as possible, in favor of more specific exceptions
   (*IOError, ValueError* etc.).
-  **Warnings:** Always use the
   ``warnings.warn(message, warning_class)``\ for warnings. These get
   redirected to ``log.warning()`` by default, but one can still use the
   standard warning-catching mechanism and custom warning classes.
-  **Debugging and Informational messages:** Always use
   ``log.info(message)`` and ``log.debug(message)``. The logging system
   uses the built-in Python logging module.

Data and Configuration
~~~~~~~~~~~~~~~~~~~~~~

-  **Storing Data:**

   -  Packages can include data in a directory named *data* inside a
      subpackage source directory as long as it is less than about 100
      kB.
   -  If the data exceeds this size, it should be hosted outside the
      source code repository, either at a third-party location on the
      internet.

Documentation and Testing
~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Docstrings:**

   -  Docstrings must be provided for all public classes, methods, and
      functions.
   -  Docstrings should follow the `numpydoc
      style <https://numpydoc.readthedocs.io/en/latest/format.html>`__
      and reStructured Text format.
   -  Write usage examples in the docstrings of all classes and
      functions whenever possible. These examples should be short and
      simple to reproduce. Users should be able to copy them verbatim
      and run them.

-  **Unit tests:** Provided for as many public methods and functions as
   possible, and should adhere to the standards set in the Testing
   Guidelines.
-  **Building Documentation:**

   -  Use sphinx to build the documentation.
   -  All extra documentation should go into a /docs sub-directory under
      the main stingray directory.

Updating and Maintaining the Changelog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stingray uses ```towncrier`` <https://pypi.org/project/towncrier/>`__
which is used to generate the ``CHANGELOG.rst`` file at the root of the
package.

As described in ``docs/changes/README.rst``, the changelog fragment
files should be added to each pull request. The changelog will be read
by users, so this description should be aimed at stingray users instead
of describing internal changes which are only relevant to the
developers. The idea is that the changelog lists all new features, API
changes, bugfixes, and so on that have been added to stingray between
versions so that a user can easily follow the changes without having to
go through the entire git log.

The towncrier tool will automatically reflow your text. You can install
towncrier and then run ``towncrier --draft`` if you want to get a
preview of how your change will look in the final release notes.

Testing Guidelines
------------------

The testing framework used by stingray is the ``pytest`` framework with ``tox``.
To run the tests, you will need to make sure you have the pytest package
(version 3.1 or later) as well as the tox tool installed.

-  Execute tests using the ``tox -e <test environment>`` command.
-  All tests should be py.test compliant: http://pytest.org/latest/.
-  Keep all tests in a /tests subdirectory under the main stingray
   directory.
-  Write one test script per module in the package.
-  Extra examples can go into an /examples folder in the main stingray
   directory, scripts that gather various data analysis tasks into
   longer procedures into a /scripts folder in the same location.

Community Guidelines
--------------------

Our Pledge
~~~~~~~~~~

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our
project and our community a harassment-free experience for everyone,
regardless of age, body size, disability, ethnicity, gender identity and
expression, level of experience, nationality, personal appearance, race,
religion, or sexual identity and orientation.

Our Standards
~~~~~~~~~~~~~

Examples of behavior that contributes to creating a positive environment
include:

-  Using welcoming and inclusive language
-  Being respectful of differing viewpoints and experiences
-  Gracefully accepting constructive criticism
-  Focusing on what is best for the community
-  Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

-  The use of sexualized language or imagery and unwelcome sexual
   attention or advances
-  Trolling, insulting/derogatory comments, and personal or political
   attacks
-  Public or private harassment
-  Publishing others’ private information, such as a physical or
   electronic address, without explicit permission
-  Other conduct which could reasonably be considered inappropriate in a
   professional setting

Our Responsibilities
~~~~~~~~~~~~~~~~~~~~

Project maintainers are responsible for clarifying the standards of
acceptable behavior and are expected to take appropriate and fair
corrective action in response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit,
or reject comments, commits, code, wiki edits, issues, and other
contributions that are not aligned to this Code of Conduct, or to ban
temporarily or permanently any contributor for other behaviors that they
deem inappropriate, threatening, offensive, or harmful.

Scope
~~~~~

This Code of Conduct applies both within project spaces and in public
spaces when an individual is representing the project or its community.
Examples of representing a project or community include using an
official project e-mail address, posting via an official social media
account, or acting as an appointed representative at an online or
offline event. Representation of a project may be further defined and
clarified by project maintainers.

Enforcement
~~~~~~~~~~~

Instances of abusive, harassing, or otherwise unacceptable behavior may
be reported by contacting the project team at any of our personal email
addresses or through private Slack communication. The project team will
review and investigate all complaints, and will respond in a way that it
deems appropriate to the circumstances. The project team is obligated to
maintain confidentiality with regard to the reporter of an incident.
Further details of specific enforcement policies may be posted
separately.

Project maintainers who do not follow or enforce the Code of Conduct in
good faith may face temporary or permanent repercussions as determined
by other members of the project’s leadership.

Attribution
~~~~~~~~~~~

This Code of Conduct is adapted from the `Contributor
Covenant <http://contributor-covenant.org>`__, version 1.4, available at
`http://contributor-covenant.org/version/1/4 <http://contributor-covenant.org/version/1/4/>`__
