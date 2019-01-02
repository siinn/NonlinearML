# Setting conda environment:

- Create conda environment

    conda env create -f environment.yml
    source activate asset_growth

- Set environment variable

    cd $CONDA_PREFIX
    mkdir -p ./etc/conda/activate.d
    open ./etc/conda/activate.d/env_vars.sh

    Add the following lines
    #!/bin/sh
    export PYSPARK_PYTHON= "conda PREFIX" + "/bin/python"
    export PYSPARK_DRIVER_PYTHON= "conda PREFIX" + "/bin/python"

# Setting Spark configuration

- Spark configuration

    open /dsvm/tools/spark/current/conf/spark-defaults.conf

    Add the following lines depending on system memory.

    spark.driver.memory              100g
    spark.driver.maxResultSize       100g

# Change system limits

- Limit on open files

    open /etc/security/limits.conf

    Add the following lines.

    * soft  nofile  500000
    * hard  nofile  500000
