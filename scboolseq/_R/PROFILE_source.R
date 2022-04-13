#' profile_binarization suite
#'
#' parallel implementation of compute_criteria()
#' implemented by Gustavo Magaña López, as part
#' of his M1 BIBS course at Université Paris-Saclay.
#'
#' The rest of the functions found within this script are either
#' verbatim copies or adaptations of the work performed by
#' the following researchers :
#'
#' Beal, Jonas
#' Montagud, Arnau
#' Traynard, Pauline
#' Barillot, Emmanuel
#' Calzone, Laurence
#'
#' at the Computational Systems Biology of Cancer group at Institut Curie
#' website : https://sysbio.curie.fr/
#'   email :  contact-sysbio@curie.fr
#'
#' This is the repository containing the original implementation in
#' Rmarkdown notebooks :
#'
#' https://github.com/sysbio-curie/PROFILE
#'

# Declare CRAN dependencies :
r_dependencies <- c(
  "mclust", "diptest", "moments", # Used for statistic calculation
  "magrittr", "tidyr", "dplyr", "tibble", # Used for data formatting + the pipe
  "bigmemory", "doSNOW", "foreach", # Used to improve parallel performance
  "glue" # Easy string interpolation in R
)

# load dependencies
..pass <- sapply(r_dependencies, library, character.only = TRUE)


random_words <- function(n = 1, word_length = 5) {
  #' Create n random words of length 'word_length'
  a <- do.call(paste0, replicate(word_length, sample(LETTERS, n, TRUE), FALSE))
  paste0(a, sprintf("%04d", sample(9999, n, TRUE)), sample(LETTERS, n, TRUE))
}

split_in_n <- function(x, n) {
  #' Split a vector in n pieces
  suppressWarnings(split(x, 1:n))
}

gaussian_mixture_from_data <- function(dataset) {
  #' This call is repeated many times around this codebase
  #' This is basically a wrapper for
  #' mclust::Mclust(na.omit(dataset), G = 2, modelNames = "E", verbose = FALSE)
  mclust::Mclust(na.omit(dataset), G = 2, modelNames = "E", verbose = FALSE)
}

BI <- function(mc) {
  #' BI : Bimodality Index
  #' Estimate
  #' Function to compute the Bimodality Index described in Wang et al. (2009)
  #' x <- dataset
  #' mc <- mclust::Mclust(na.omit(x), G = 2, modelNames = "E", verbose = FALSE)

  if (is.null(mc)) {
    b_i <- NA
  } else {
    sigma <- sqrt(mc$parameters$variance$sigmasq)
    delta <- abs(diff(mc$parameters$mean)) / sigma
    p_i <- mc$parameters$pro[1]
    b_i <- delta * sqrt(p_i * (1 - p_i))
  }
  b_i
}

OSclass <- function(exp_dataset, ref_dataset = exp_dataset) {
  #' Function to binarise the tails of the distribution
  #' Based on inter-quartile range (IQR)
  #' similar to methods described in teh outlier-sum statistic
  #' (Tibshirani and Hastie, 2007).
  #' Can be called with a reference dataset

  classif <- rep(NA, length(exp_dataset))
  q25 <- quantile(ref_dataset, 0.25, na.rm = T)
  q75 <- quantile(ref_dataset, 0.75, na.rm = T)
  IQR <- q75 - q25 # Inter-Quartile Range

  classif[exp_dataset > IQR + q75] <- 1
  classif[exp_dataset < q25 - IQR] <- 0
  return(classif)
}

BIMclass <- function(exp_dataset, ref_dataset = exp_dataset) {
  #' Function to to binarise bimodal distributions
  #' based on a 2-modes gaussian mixture model (with equal variances).
  #' Can be called with a reference dataset
  mc <- mclust::Mclust(
    na.omit(ref_dataset),
    modelNames = "E",
    G = 2, verbose = FALSE
  )
  classif <- rep(NA, length(exp_dataset))
  if (diff(mc$parameters$mean) > 0) {
    thresh_down <- max(mc$data[mc$classification == 1 & mc$uncertainty <= 0.05])
    thresh_up <- min(mc$data[mc$classification == 2 & mc$uncertainty <= 0.05])
    classif[exp_dataset <= thresh_down] <- 0
    classif[exp_dataset >= thresh_up] <- 1
  } else if (diff(mc$parameters$mean) < 0) {
    thresh_down <- max(mc$data[mc$classification == 2 & mc$uncertainty <= 0.05])
    thresh_up <- min(mc$data[mc$classification == 1 & mc$uncertainty <= 0.05])
    classif[exp_dataset <= thresh_down] <- 0
    classif[exp_dataset >= thresh_up] <- 1
  }
  return(classif)
}

norm_fun_lin <- function(xdat, reference = xdat) {
  #' Function for normalisation of zero-inflated data
  x_proc <- (xdat - quantile(reference, 0.01, na.rm = T)) /
    quantile(xdat - quantile(reference, 0.01, na.rm = T), 0.99, na.rm = T)
  x_proc[x_proc < 0] <- 0
  x_proc[x_proc > 1] <- 1
  x_proc
}

norm_fun_sig <- function(xdat, reference = xdat) {
  #' Function for normalisation of unimodal data
  xdat <- xdat - median(reference, na.rm = T)
  lambda <- log(3) / mad(reference, na.rm = T)
  transformation <- function(x) {
    y <- 1 / (1 + exp(-lambda * x))
    y
  }
  transformation(xdat)
}

norm_fun_bim <- function(xdat, reference = xdat) {
  #' Function for normalisation of unimodal data
  not_na_xdat <- !is.na(xdat)
  not_na_ref <- !is.na(reference)
  mc <- mclust::Mclust(
    reference[not_na_ref],
    modelNames = "E",
    G = 2,
    verbose = FALSE
  )
  pred <- mclust::predict.Mclust(mc, xdat[not_na_xdat])
  normalization <- rep(NA, length(xdat))
  if (diff(mc$parameters$mean) > 0) {
    normalization[not_na_xdat] <- pred$z[, 2]
  } else if (diff(mc$parameters$mean) < 0) {
    normalization[not_na_xdat] <- pred$z[, 1]
  }
  normalization
}


criteria_iter <- function(columns, data, genes,
                          mask_zero_entries = FALSE,
                          unimodal_margin_quantile = 0.25) {
  #' Compute criteria for a subset of genes
  #'
  #' Data should be generated calling
  #' bigmemory::attach.big.matrix()
  #' on a big_matrix_descriptor.
  #'
  #' This is an auxiliary function intended
  #' to be called by compute_criteria
  #' and not directly by the user.

  criterix <- foreach::foreach(i = columns, .combine = rbind) %do% {
    x <- na.omit(unlist(data[, i]))
    criteria.iter <- list(
      # original criteria
      Gene = genes[i], Dip = NA, BI = NA, Kurtosis = NA,
      DropOutRate = NA, MeanNZ = NA,
      DenPeak = NA, Amplitude = max(x) - min(x),
      # added criteria
      gaussian_prob1 = NA,
      gaussian_prob2 = NA,
      gaussian_mean1 = NA,
      gaussian_mean2 = NA,
      gaussian_variance = NA,
      mean = NA,
      variance = NA,
      unimodal_margin_quantile = NA,
      unimodal_low_quantile = NA,
      unimodal_high_quantile = NA,
      IQR = NA,
      q50 = NA,
      bim_thresh_down = NA,
      bim_thresh_up = NA
    )

    if (criteria.iter$Amplitude != 0) {
      mc <- gaussian_mixture_from_data(x)

      # add original criteria
      criteria.iter$DropOutRate <- sum(x == 0) / length(x)
      den <- density(x, na.rm = T)
      criteria.iter$DenPeak <- den$x[which.max(den$y)]
      ## new approach (better estimation ?)
      if (mask_zero_entries) {
        x <- x[x > 0]
      }
      # continue adding the original criteria
      criteria.iter$Dip <- diptest::dip.test(x)$p.value
      criteria.iter$BI <- BI(mc)
      criteria.iter$Kurtosis <- moments::kurtosis(x) - 3
      criteria.iter$MeanNZ <- sum(x) / sum(x != 0)

      # add enhanced criteria (used for generation)
      criteria.iter$unimodal_margin_quantile <- unimodal_margin_quantile
      criteria.iter$unimodal_low_quantile <-
        quantile(x, unimodal_margin_quantile)
      criteria.iter$unimodal_high_quantile <-
        quantile(x, 1.0 - unimodal_margin_quantile)
      criteria.iter$IQR <- IQR(x)
      criteria.iter$q50 <- quantile(x, 0.50)
      # criteria.iter$zero_inf_thresh <-
      #  criteria.iter$IQR + criteria.iter$q75
      ## parameters for bimodal genes :
      criteria.iter$gaussian_prob1 <- mc$parameters$pro[1]
      criteria.iter$gaussian_prob2 <- mc$parameters$pro[2]
      criteria.iter$gaussian_mean1 <- mc$parameters$mean[1]
      criteria.iter$gaussian_mean2 <- mc$parameters$mean[2]
      criteria.iter$gaussian_variance <- mc$parameters$variance$sigmasq
      # save parameters for python-side binarisation
      .delta <- as.integer(diff(mc$parameters$mean) > 0)
      .alpha <- 1
      .beta <- 2
      .down <- as.integer(.delta * .alpha + (1 - .delta) * .beta)
      .up <- as.integer((1 - .delta) * .alpha + .delta * .beta)
      criteria.iter$bim_thresh_down <-
        max(mc$data[mc$classification == .down & mc$uncertainty <= 0.05])
      criteria.iter$bim_thresh_up <-
        min(mc$data[mc$classification == .up & mc$uncertainty <= 0.05])
      ## parameters for unimodal genes :
      criteria.iter$mean <- mean(x)
      criteria.iter$variance <- var(x)
    }

    as.data.frame(criteria.iter)
  }
  criterix
}

compute_criteria <- function(exp_dataset, n_threads,
                             dor_threshold = 0.95, # change this to 0.85 ?
                             mask_zero_entries = FALSE,
                             unimodal_margin_quantile = 0.25,
                             descriptor_filename = NULL) {
  #' Function used to compute all statistical tools and criteria
  #' needed to perform the classification of distributions
  #' in the following categories:
  #'  * discarded
  #'  * zero-inflated
  #'  * unimodal
  #'  * bimodal
  #'

  exp_dataset %<>% tibble::rownames_to_column("individual_id")

  .remove_descriptor <- FALSE
  if (is.null(descriptor_filename)) {
    descriptor_filename <-
      glue::glue("SCBOOLSEQ_backing_file_{random_words()}_{date()}")
    .remove_descriptor <- TRUE
  }

  backing_file <- glue::glue("{descriptor_filename}.bin")
  descriptor_file <- glue::glue("{descriptor_filename}.desc")

  # .col_names <- colnames(exp_dataset)
  genes <- exp_dataset %>%
    dplyr::select(-individual_id) %>%
    colnames()


  tryCatch(
    {
      parallel_cluster <-
        snow::makeSOCKcluster(names = rep("localhost", n_threads))
      doSNOW::registerDoSNOW(parallel_cluster)
      big_exp_dataset <- exp_dataset %>%
        dplyr::select(-individual_id) %>%
        as.data.frame() %>%
        bigmemory::as.big.matrix(
          type = "double", separated = FALSE,
          backingfile = backing_file,
          descriptorfile = descriptor_file
        )

      big_exp_descriptor <- bigmemory::describe(big_exp_dataset)
      gene_iterator <- split_in_n(1:ncol(big_exp_dataset), n_threads)

      criteria <- foreach::foreach(
        i = gene_iterator, .combine = rbind, .inorder = TRUE,
        .export = c("criteria_iter", "BI", "gaussian_mixture_from_data")
      ) %dopar% {
        require(foreach)
        require(mclust)
        yy <- bigmemory::attach.big.matrix(big_exp_descriptor)
        criteria_iter(
          i, yy, genes,
          mask_zero_entries = mask_zero_entries,
          unimodal_margin_quantile = unimodal_margin_quantile
        )
      }
    },
    finally = {
      snow::stopCluster(parallel_cluster)
      if (.remove_descriptor) {
        unlink(backing_file)
        unlink(descriptor_file)
      }
    }
  )


  threshold <- median(criteria$Amplitude) / 10
  # Added `tibble` call to enable the use of dplyr operators.
  criteria <- criteria %>%
    dplyr::tibble() %>%
    dplyr::mutate(Category = ifelse(
      Amplitude < threshold | DropOutRate > dor_threshold,
      "Discarded", NA
    )) %>%
    dplyr::mutate(Category = ifelse(
      is.na(Category) &
        (BI > 1.5 & Dip < 0.05 & Kurtosis < 1),
      "Bimodal", Category
    )) %>%
    dplyr::mutate(Category = ifelse(
      is.na(Category) & DenPeak < threshold,
      "ZeroInf", Category
    )) %>%
    dplyr::mutate(Category = ifelse(is.na(Category), "Unimodal", Category))

  criteria %<>% tibble::column_to_rownames("Gene")
  criteria[genes, ]
}

# compute_criteria_sequential <- function(exp_dataset, individual_id)
# removed this function because it was superseded by compute_criteria
# if you ever want to re-implement it or re-use it, come back to this
# commit : 1fab19e7573d9ce00f4db36f42dcefdc408d6a72
# on branch main of https://github.com/bnediction/profile_binr/

binarize_exp <- function(exp_dataset, ref_dataset, ref_criteria, gene) {
  #' function to apply the proper binarization method depending
  #' on the gene expression distribution category
  #' Sorts ref_criteria rows according to exp_dataset

  .col_names <- colnames(exp_dataset)

  # Boolean flags to verify there is a reference for each
  # and every single gene of the exp_dataset
  .is_subset_of_ref_dataset <-
    Reduce(`&`, (.col_names %in% colnames(ref_dataset)))
  .is_subset_of_criteria <-
    Reduce(`&`, (.col_names %in% rownames(ref_criteria)))
  stopifnot(.is_subset_of_criteria, .is_subset_of_ref_dataset)

  # Sort the criteria according to the order of appearance in exp_dataset
  ref_criteria <- ref_criteria[.col_names, ]
  ref_dataset <- ref_dataset[, .col_names]

  if (!missing(gene)) {
    stopifnot(gene %in% rownames(ref_criteria), gene %in% colnames(ref_dataset))

    gene_cat <- ref_criteria[gene, "Category"]
    x <- unlist(dplyr::select(exp_dataset, gene))
    x_ref <- unlist(dplyr::select(ref_dataset, gene))

    if (gene_cat == "Discarded") {
      gene_bin <- rep(NA, length(x))
    } else if (gene_cat == "Bimodal") {
      gene_bin <- BIMclass(x, x_ref)
    } else {
      gene_bin <- OSclass(x, x_ref)
    }

    names(gene_bin) <- colnames(exp_dataset)

    return(gene_bin)
  } else {
    if (dim(exp_dataset)[2] != dim(ref_criteria)[1]) {
      stop("Different number of genes")
    }

    # these vectors should match the order
    logi_dis <- ref_criteria$Category == "Discarded"
    logi_OS <- ref_criteria$Category == "Unimodal" | ref_criteria$Category ==
      "ZeroInf"
    logi_bim <- ref_criteria$Category == "Bimodal"

    exp_dataset[, logi_dis] <- sapply(exp_dataset[, logi_dis], function(x) {
      rep(
        NA,
        length(x)
      )
    })
    exp_dataset[, logi_OS] <- mapply(
      OSclass, as.data.frame(exp_dataset[, logi_OS]),
      as.data.frame(ref_dataset[, logi_OS])
    )
    exp_dataset[, logi_bim] <- mapply(
      BIMclass, as.data.frame(exp_dataset[, logi_bim]),
      as.data.frame(ref_dataset[, logi_bim])
    )

    colnames(exp_dataset) <- .col_names
    return(exp_dataset)
  }
}


normalize_exp <- function(exp_dataset, ref_dataset, ref_criteria, gene) {
  #' Normalise gene expression dataset
  #' using a reference dataset and reference
  #' criteria.

  .col_names <- colnames(exp_dataset)

  # Boolean flags to verify there is a reference for each and every single gene of
  # the exp_dataset
  .is_subset_of_ref <- Reduce(`&`, (.col_names %in% colnames(ref_dataset)))
  .is_subset_of_criteria <- Reduce(`&`, (.col_names %in% rownames(ref_criteria)))
  stopifnot(.is_subset_of_criteria, .is_subset_of_ref)

  ref_criteria <- ref_criteria[.col_names, ]
  ref_dataset <- ref_dataset[, .col_names]

  if (!missing(gene)) {
    stopifnot(gene %in% rownames(ref_criteria), gene %in% colnames(ref_dataset))

    gene_cat <- ref_criteria[gene, "Category"]
    x <- unlist(dplyr::select(exp_dataset, gene))
    x_ref <- unlist(dplyr::select(ref_dataset, gene))

    if (gene_cat == "Discarded") {
      gene_bin <- rep(NA, length(x))
    } else if (gene_cat == "Bimodal") {
      gene_bin <- norm_fun_bim(x, x_ref)
    } else if (gene_cat == "Unimodal") {
      gene_bin <- norm_fun_sig(x, x_ref)
    } else {
      gene_bin <- norm_fun_lin(x, x_ref)
    }

    names(gene_bin) <- colnames(exp_dataset)

    return(gene_bin)
  } else {
    if (dim(exp_dataset)[2] != dim(ref_criteria)[1]) {
      stop("Different number of genes")
    }

    logi_dis <- ref_criteria$Category == "Discarded"
    logi_uni <- ref_criteria$Category == "Unimodal"
    logi_zero <- ref_criteria$Category == "ZeroInf"
    logi_bim <- ref_criteria$Category == "Bimodal"

    exp_dataset[, logi_dis] <- sapply(exp_dataset[, logi_dis], function(x) {
      rep(
        NA,
        length(x)
      )
    })
    exp_dataset[, logi_uni] <- mapply(norm_fun_sig, as.data.frame(exp_dataset[
      ,
      logi_uni
    ]), as.data.frame(ref_dataset[, logi_uni]))
    exp_dataset[, logi_zero] <- mapply(norm_fun_lin, as.data.frame(exp_dataset[
      ,
      logi_zero
    ]), as.data.frame(ref_dataset[, logi_zero]))
    exp_dataset[, logi_bim] <- mapply(norm_fun_bim, as.data.frame(exp_dataset[
      ,
      logi_bim
    ]), as.data.frame(ref_dataset[, logi_bim]))

    return(exp_dataset)
  }
}