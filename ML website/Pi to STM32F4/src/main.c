/*
    HAL Initialization and Clock Configuration:
    The program starts by initializing the HAL and configuring the system clock. 
    The SystemClock_Config() function provided here is a basic example using the HSI oscillator and PLL. 
    You may need to adjust this to match your board's specifications.

    UART Initialization:
    MX_USART2_UART_Init() sets up USART2 with a baud rate of 115200. 
    Make sure that USART2 is available on your board and that your wiring 
    (including any virtual COM port) is correct.

    Redirecting printf:
    The _write() function sends output from printf to USART2. 
    This is useful if you’re using a serial terminal or debugger that captures UART output. 

    Receiving and Processing Data:
    In the main loop, the firmware waits for incoming data on UART. 
    Once received, the data is parsed by processCoordinates(), 
    which uses sscanf to extract the x and y coordinates and a name. 
    If parsing is successful, the data is printed via printf.
*/

#include "stm32f4xx_hal.h"
#include <stdio.h>
#include <string.h>

// Global UART handle for USART2
UART_HandleTypeDef huart2;

// Function prototypes
void SystemClock_Config(void);
static void MX_USART2_UART_Init(void);

// Redirect printf output to UART2 (this function is used by printf)
int _write(int file, char *data, int len) {
    HAL_UART_Transmit(&huart2, (uint8_t*)data, len, HAL_MAX_DELAY);
    return len;
}

/**
  * @brief  Process the received coordinate string.
  *         Expected format: "x,y,name" where x and y are integers and name is a string.
  * @param  buffer: Pointer to the received string.
  * @retval None
  */
void processCoordinates(char *buffer) {
    int x, y;
    char name[20] = {0};
    // Attempt to parse the coordinates and a name from the buffer
    if (sscanf(buffer, "%d,%d,%19s", &x, &y, name) == 3) {
        // Print the received data to the console
        printf("Received: x = %d, y = %d, name = %s\r\n", x, y, name);
    } else {
        printf("Failed to parse coordinates from: %s\r\n", buffer);
    }
}

int main(void) {
    // Initialize the HAL Library; it must be the first function to be executed before the use of any HAL function.
    HAL_Init();
    
    // Configure the system clock
    SystemClock_Config();
    
    // Initialize USART2 for UART communication
    MX_USART2_UART_Init();
    
    // Buffer to hold received data
    char rxBuffer[50] = {0};

    // Main loop: receive coordinate strings via UART and process them.
    while (1) {
        // Receive data over UART (blocking call)
        if (HAL_UART_Receive(&huart2, (uint8_t*)rxBuffer, sizeof(rxBuffer) - 1, HAL_MAX_DELAY) == HAL_OK) {
            // Remove any newline character and ensure the string is null-terminated
            rxBuffer[strcspn(rxBuffer, "\n")] = 0;
            processCoordinates(rxBuffer);
            // Clear the buffer for the next message
            memset(rxBuffer, 0, sizeof(rxBuffer));
        }
    }
}

/**
  * @brief Initialize USART2.
  * @note  Adjust the UART settings (baud rate, word length, etc.) to suit your board.
  * @retval None
  */
static void MX_USART2_UART_Init(void) {
    huart2.Instance = USART2;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart2) != HAL_OK) {
        // Initialization error: stay here
        while (1);
    }
}

/**
  * @brief System Clock Configuration
  *        This example configures the system clock to run from the PLL using the internal HSI oscillator.
  *        You may need to adjust these settings based on your specific hardware.
  * @retval None
  */
void SystemClock_Config(void) {
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    // Enable power control clock and configure voltage scaling
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    // Configure the HSI oscillator and enable the PLL
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 16;
    RCC_OscInitStruct.PLL.PLLN = 336;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4; // Adjust if necessary
    RCC_OscInitStruct.PLL.PLLQ = 7;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        // Oscillator configuration error: stay here
        while (1);
    }
    // Configure clocks for AHB and APB buses
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                  RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) {
        // Clock configuration error: stay here
        while (1);
    }
}
