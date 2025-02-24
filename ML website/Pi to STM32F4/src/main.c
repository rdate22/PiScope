#include "stm32f4xx_hal.h"

#define LED_PIN GPIO_PIN_13  // LD3 is on PD13
#define LED_PORT GPIOD       // LD3 is on GPIOD

void SystemClock_Config(void);
static void MX_GPIO_Init(void);

int main(void) {
    HAL_Init();              // Initialize the HAL Library
    SystemClock_Config();     // Configure the system clock
    MX_GPIO_Init();          // Initialize GPIO

    while (1) {
        HAL_GPIO_TogglePin(LED_PORT, LED_PIN); // Toggle LD3 (PD13)
        HAL_Delay(500);  // 500ms delay
    }
}

// Initialize GPIO for onboard LED
void MX_GPIO_Init(void) {
    __HAL_RCC_GPIOD_CLK_ENABLE(); // Enable GPIOD clock

    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = LED_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LED_PORT, &GPIO_InitStruct);
}

// System Clock Configuration (You can generate this using STM32CubeMX)
void SystemClock_Config(void) {
    // Default configuration (replace with CubeMX-generated settings if needed)
}
