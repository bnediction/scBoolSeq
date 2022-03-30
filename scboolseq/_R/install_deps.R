
# Try to load all dependencies :
r_dependencies <- c("mclust", "diptest", "moments", "magrittr", "tidyr", "dplyr", 
    "tibble", "bigmemory", "doSNOW", "foreach", "glue")
.deps_loaded <- sapply(r_dependencies, require, character.only=T)

# Install those that are not available :
.not_installed <- Filter(function(x) !x, .deps_loaded) 
.to_install <- names(.not_installed)
install.packages(.to_install)
