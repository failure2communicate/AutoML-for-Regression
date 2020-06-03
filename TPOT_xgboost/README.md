# OS Packages Documentation  
## TPOT Automated Machine Learning Pipeline (for regression)

### 1.	Model details
#### Input type: 
JSON

#### Input Description:  
Features used by the model to make predictions. For example: {
“Feature1”:  12,
“Feature2”:  222,
.
.
“FeatureN”:  110
}

#### Output Description: 
JSON with list of predictions: {
  "predictions" : "[12, 12, 2, 354, 12, 2]
}

#### Language: 
Python 3.6 


### 2.	Package details:

#### Description: 
Please train this ML Package before deploying it as it will not return anything otherwise.  
TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming. TPOT will automate the most tedious part of machine learning by intelligently exploring thousands of possible pipelines to find the best one for your data. Once TPOT is finished searching (or you get tired of waiting), it provides you with the Python code for the best pipeline it found so you can tinker with the pipeline from there. TPOT is built on top of scikit-learn, so all the code it generates should look familiar to scikit-learn users.

This version of TPOT uses only XGBoost and the standard set of pre-processing methods to optimize a machine learning pipeline.

#### Pipelines: 
This package can run with the three type of pipelines: Full pipeline, Training Pipeline and Evaluation Pipeline.

#### Dataset format: 
This ML Package will look for csv files in your dataset (not in subdirectories).

    •	csv files: First row of the data must contain the header/column names. All columns, except for the target column, must be numerical (int, float). The model is not able perform feature encoding.

#### Environment variables: 
•	“train_time”: time to tun the pipeline (in minutes). The longer the train time the better chances TPOT has at finding a good model. (default: 2)

•	“target_column”: name of the target column (default: “target”)

•	“scoring”: TPOT makes use of sklearn.model_selection.cross_val_score for evaluating pipelines, and as such offers the same support for scoring functions (default: “neg_mean_squared_error”). The following built-in scoring functions can be used: {'neg_median_absolute_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'}. Custom scoring functions can be defined as well: https://epistasislab.github.io/tpot/using/#scoring-functions

•	"keep_training": Typical TPOT runs will take hours to days to finish (unless it's a small dataset), but you can always interrupt the run partway through and see the best results so far. If keep_training is set to True, TPOT will continue the training where it left of.

#### Artifacts:
TPOT exports the corresponding Python code for the optimized pipeline to a python file called “TPOT_pipeline.py”. Once the code finishes running, “TPOT_pipeline.py” will contain the Python code for the optimized pipeline.

#### Paper: 
The model is based on a publication entitled "Scaling tree-based automated machine learning to biomedical big data with a feature set selector." from Trang T. Le, Weixuan Fu and Jason H. Moore (2020) and "Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science." from Randal S. Olson, Nathan Bartley, Ryan J. Urbanowicz, and Jason H. Moore.

The publication can be found here:
-	https://academic.oup.com/bioinformatics/article/36/1/250/5511404
-	http://dl.acm.org/citation.cfm?id=2908918

#### Documentation:
https://epistasislab.github.io/tpot/

### License
    GNU Lesser General Public License v3.0
    Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
    Everyone is permitted to copy and distribute verbatim copies
    of this license document, but changing it is not allowed.


    This version of the GNU Lesser General Public License incorporates
    the terms and conditions of version 3 of the GNU General Public
    License, supplemented by the additional permissions listed below.

    0. Additional Definitions.

    As used herein, "this License" refers to version 3 of the GNU Lesser
    General Public License, and the "GNU GPL" refers to version 3 of the GNU
    General Public License.

    "The Library" refers to a covered work governed by this License,
    other than an Application or a Combined Work as defined below.

    An "Application" is any work that makes use of an interface provided
    by the Library, but which is not otherwise based on the Library.
    Defining a subclass of a class defined by the Library is deemed a mode
    of using an interface provided by the Library.

    A "Combined Work" is a work produced by combining or linking an
    Application with the Library.  The particular version of the Library
    with which the Combined Work was made is also called the "Linked
    Version".

    The "Minimal Corresponding Source" for a Combined Work means the
    Corresponding Source for the Combined Work, excluding any source code
    for portions of the Combined Work that, considered in isolation, are
    based on the Application, and not on the Linked Version.

    The "Corresponding Application Code" for a Combined Work means the
    object code and/or source code for the Application, including any data
    and utility programs needed for reproducing the Combined Work from the
    Application, but excluding the System Libraries of the Combined Work.

    1. Exception to Section 3 of the GNU GPL.

    You may convey a covered work under sections 3 and 4 of this License
    without being bound by section 3 of the GNU GPL.

    2. Conveying Modified Versions.

    If you modify a copy of the Library, and, in your modifications, a
    facility refers to a function or data to be supplied by an Application
    that uses the facility (other than as an argument passed when the
    facility is invoked), then you may convey a copy of the modified
    version:

    a) under this License, provided that you make a good faith effort to
    ensure that, in the event an Application does not supply the
    function or data, the facility still operates, and performs
    whatever part of its purpose remains meaningful, or

    b) under the GNU GPL, with none of the additional permissions of
    this License applicable to that copy.

    3. Object Code Incorporating Material from Library Header Files.

    The object code form of an Application may incorporate material from
    a header file that is part of the Library.  You may convey such object
    code under terms of your choice, provided that, if the incorporated
    material is not limited to numerical parameters, data structure
    layouts and accessors, or small macros, inline functions and templates
    (ten or fewer lines in length), you do both of the following:

    a) Give prominent notice with each copy of the object code that the
    Library is used in it and that the Library and its use are
    covered by this License.

    b) Accompany the object code with a copy of the GNU GPL and this license
    document.

    4. Combined Works.

    You may convey a Combined Work under terms of your choice that,
    taken together, effectively do not restrict modification of the
    portions of the Library contained in the Combined Work and reverse
    engineering for debugging such modifications, if you also do each of
    the following:

    a) Give prominent notice with each copy of the Combined Work that
    the Library is used in it and that the Library and its use are
    covered by this License.

    b) Accompany the Combined Work with a copy of the GNU GPL and this license
    document.

    c) For a Combined Work that displays copyright notices during
    execution, include the copyright notice for the Library among
    these notices, as well as a reference directing the user to the
    copies of the GNU GPL and this license document.

    d) Do one of the following:

        0) Convey the Minimal Corresponding Source under the terms of this
        License, and the Corresponding Application Code in a form
        suitable for, and under terms that permit, the user to
        recombine or relink the Application with a modified version of
        the Linked Version to produce a modified Combined Work, in the
        manner specified by section 6 of the GNU GPL for conveying
        Corresponding Source.

        1) Use a suitable shared library mechanism for linking with the
        Library.  A suitable mechanism is one that (a) uses at run time
        a copy of the Library already present on the user's computer
        system, and (b) will operate properly with a modified version
        of the Library that is interface-compatible with the Linked
        Version.

    e) Provide Installation Information, but only if you would otherwise
    be required to provide such information under section 6 of the
    GNU GPL, and only to the extent that such information is
    necessary to install and execute a modified version of the
    Combined Work produced by recombining or relinking the
    Application with a modified version of the Linked Version. (If
    you use option 4d0, the Installation Information must accompany
    the Minimal Corresponding Source and Corresponding Application
    Code. If you use option 4d1, you must provide the Installation
    Information in the manner specified by section 6 of the GNU GPL
    for conveying Corresponding Source.)

    5. Combined Libraries.

    You may place library facilities that are a work based on the
    Library side by side in a single library together with other library
    facilities that are not Applications and are not covered by this
    License, and convey such a combined library under terms of your
    choice, if you do both of the following:

    a) Accompany the combined library with a copy of the same work based
    on the Library, uncombined with any other library facilities,
    conveyed under the terms of this License.

    b) Give prominent notice with the combined library that part of it
    is a work based on the Library, and explaining where to find the
    accompanying uncombined form of the same work.

    6. Revised Versions of the GNU Lesser General Public License.

    The Free Software Foundation may publish revised and/or new versions
    of the GNU Lesser General Public License from time to time. Such new
    versions will be similar in spirit to the present version, but may
    differ in detail to address new problems or concerns.

    Each version is given a distinguishing version number. If the
    Library as you received it specifies that a certain numbered version
    of the GNU Lesser General Public License "or any later version"
    applies to it, you have the option of following the terms and
    conditions either of that published version or of any later version
    published by the Free Software Foundation. If the Library as you
    received it does not specify a version number of the GNU Lesser
    General Public License, you may choose any version of the GNU Lesser
    General Public License ever published by the Free Software Foundation.

    If the Library as you received it specifies that a proxy can decide
    whether future versions of the GNU Lesser General Public License shall
    apply, that proxy's public statement of acceptance of any version is
    permanent authorization for you to choose that version for the
    Library.
 

